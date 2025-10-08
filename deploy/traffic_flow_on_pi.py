import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import pandas as pd
from collections import defaultdict
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt

# 1. Tải model và video (sử dụng wget thay vì gdown nếu cần)
# Giả sử bạn đã có file best.pt và vehicle_counting.mp4 trong thư mục hiện tại

SOURCE_VIDEO_PATH = "vehicle_counting.mp4"
MODEL_PATH = "weights/best.pt"

# 2. Load model
model = YOLO(MODEL_PATH)

# 3. Cấu hình classes
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ["motorbike", "car", "bus", "truck"]
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

print("Selected IDs:", SELECTED_CLASS_IDS, [CLASS_NAMES_DICT[i] for i in SELECTED_CLASS_IDS])

# 4. Khởi tạo ByteTrack với cài đặt nhẹ hơn cho Raspberry Pi
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.3,  # Tăng ngưỡng để giảm false positives
    lost_track_buffer=15,           # Giảm buffer để tiết kiệm bộ nhớ
    minimum_matching_threshold=0.7,
    frame_rate=15,                  # Giảm FPS cho Raspberry Pi
    minimum_consecutive_frames=2
)

# 5. Lấy thông tin video
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
W, H, FPS = video_info.width, video_info.height, max(1, int(round(video_info.fps or 15)))

# Giảm độ phân giải nếu cần để tăng tốc độ xử lý
if W > 640:
    SCALE_FACTOR = 640 / W
    W, H = int(W * SCALE_FACTOR), int(H * SCALE_FACTOR)
    print(f"Scaled video to: {W}x{H}")

def P(px, py):
    return (int(px * W), int(py * H))

def rect_frac(l, t, r, b):
    return [P(l, t), P(r, t), P(r, b), P(l, b)]

# 6. Định nghĩa ROI và các biến khác (giữ nguyên)
REGIONS_FRACTION = {
    "1": rect_frac(0.01, 0.28, 0.22, 0.9),
    "2": rect_frac(0.3, 0.01, 0.78, 0.22),
    "3": rect_frac(0.8, 0.22, 0.99, 0.85),
    "4": rect_frac(0.23, 0.88, 0.72, 0.99),
}

REGION_COLORS = {
    "normal": sv.Color(r=255, g=0, b=0),
    "active": sv.Color(r=0, g=255, b=0),
}

region_names = ["1","2","3","4"]
zones, zone_annots = [], []

class CustomPolygonZoneAnnotator:
    def __init__(self, zone, color, thickness=2, text_thickness=1, text_scale=1):
        self.zone = zone
        self.color = color
        self.thickness = thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale

    def annotate(self, scene, color_override=None):
        color = color_override if color_override else self.color
        cv2.polylines(scene, [self.zone.polygon], True, color.as_bgr(), self.thickness)
        return scene

for name in region_names:
    poly = np.array(REGIONS_FRACTION[name], dtype=np.int32)
    zone = sv.PolygonZone(polygon=poly)
    zones.append(zone)
    zone_annot = CustomPolygonZoneAnnotator(zone=zone, color=REGION_COLORS["normal"])
    zone_annots.append(zone_annot)

# 7. Các biến để đếm và theo dõi
region_total_counts = [defaultdict(int) for _ in zones]
region_present_ids = [set() for _ in zones]
region_active_frames = [0 for _ in zones]
ACTIVE_DURATION = 30

flow_data = []
current_flow_counts = [defaultdict(int) for _ in zones]
flow_start_time = 0
FLOW_INTERVAL = 60  # 60 giây

CLASS_ID_TO_NAME = {cid: model.model.names[cid] for cid in SELECTED_CLASS_IDS}
LOG_ROWS = []

# 8. Annotators với cài đặt đơn giản hóa
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)  # Giảm trace length
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)

# 9. Hàm callback đã được tối ưu
def callback_regions(frame: np.ndarray, index: int) -> np.ndarray:
    global flow_start_time, current_flow_counts

    # Resize frame để tăng tốc độ xử lý
    if frame.shape[1] != W or frame.shape[0] != H:
        frame = cv2.resize(frame, (W, H))
    
    current_time = index / FPS

    # Inference với model - sử dụng kích thước nhỏ hơn
    results = model(frame, imgsz=320, verbose=False)[0]  # Giảm kích thước ảnh đầu vào
    det = sv.Detections.from_ultralytics(results)
    det = det[np.isin(det.class_id, SELECTED_CLASS_IDS)]
    det = byte_tracker.update_with_detections(det)

    # Vẽ annotations
    annotated = frame.copy()
    labels = [f"#{tid} {model.model.names[c]}"
              for p, c, tid in zip(det.confidence, det.class_id, det.tracker_id)]
    
    annotated = trace_annotator.annotate(annotated, det)
    annotated = box_annotator.annotate(annotated, det)
    annotated = label_annotator.annotate(annotated, det, labels=labels)

    # Xử lý từng vùng
    for zi, zone in enumerate(zones):
        inside_mask = zone.trigger(det)
        new_entries_this_frame = False

        # Kiểm tra đối tượng mới vào vùng
        for k, inside in enumerate(inside_mask):
            tid = det.tracker_id[k]
            if tid is None:
                continue

            if inside and tid not in region_present_ids[zi]:
                region_present_ids[zi].add(tid)
                cid = det.class_id[k]
                region_total_counts[zi][cid] += 1
                current_flow_counts[zi][cid] += 1
                new_entries_this_frame = True
                region_active_frames[zi] = ACTIVE_DURATION

        # Xử lý lưu lượng theo khoảng thời gian
        if current_time - flow_start_time >= FLOW_INTERVAL:
            flow_record = {
                "timestamp": current_time,
                "time_str": str(timedelta(seconds=int(current_time))),
                "interval": FLOW_INTERVAL
            }

            for zi, region_name in enumerate(region_names):
                total_flow = sum(current_flow_counts[zi].values())
                flow_record[f"region_{region_name}_total"] = total_flow
                flow_record[f"region_{region_name}_flow_per_min"] = total_flow * (60 / FLOW_INTERVAL)

                for cid in SELECTED_CLASS_IDS:
                    class_name = CLASS_ID_TO_NAME[cid]
                    count = current_flow_counts[zi][cid]
                    flow_record[f"region_{region_name}_{class_name}"] = count
                    flow_record[f"region_{region_name}_{class_name}_per_min"] = count * (60 / FLOW_INTERVAL)

            flow_data.append(flow_record)
            current_flow_counts = [defaultdict(int) for _ in zones]
            flow_start_time = current_time

        # Giảm counter active frames
        if region_active_frames[zi] > 0:
            region_active_frames[zi] -= 1

        # Vẽ polygon và thông tin
        current_color = REGION_COLORS["active"] if region_active_frames[zi] > 0 else REGION_COLORS["normal"]
        annotated = zone_annots[zi].annotate(annotated, color_override=current_color)

        # Vẽ thông tin thống kê
        poly = np.array(REGIONS_FRACTION[region_names[zi]], dtype=np.float32)
        cx, cy = poly.mean(axis=0).astype(int)
        occ = int(np.sum(inside_mask))
        
        current_flow_per_min = sum(current_flow_counts[zi].values()) * (60 / FLOW_INTERVAL)
        text_color = (0, 255, 0) if region_active_frames[zi] > 0 else (0, 0, 255)

        # Vẽ text với font nhỏ hơn
        cv2.putText(annotated, f"R{region_names[zi]} Occ:{occ}", (cx-120, cy+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
        
        cv2.putText(annotated, f"Flow: {current_flow_per_min:.1f}/m", (cx-120, cy+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

        # Ghi log
        row = {
            "frame": index,
            "time_s": round(index / FPS, 3),
            "region": region_names[zi],
            "occupancy": occ,
            "current_flow_per_min": current_flow_per_min,
            "new_entry": int(new_entries_this_frame)
        }
        for cid in SELECTED_CLASS_IDS:
            row[f"count_{CLASS_ID_TO_NAME[cid]}"] = int(region_total_counts[zi][cid])
        LOG_ROWS.append(row)

    return annotated

# 10. Xử lý video
TARGET_VIDEO_PATH = "vehicle_counting_4regions_pi.mp4"

print("Bắt đầu xử lý video...")
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback_regions
)

# 11. Lưu kết quả
csv_path = "region4_counts_with_flow.csv"
pd.DataFrame(LOG_ROWS).to_csv(csv_path, index=False)

if len(flow_data) > 0:
    flow_df = pd.DataFrame(flow_data)
    flow_csv_path = "traffic_flow_analysis.csv"
    flow_df.to_csv(flow_csv_path, index=False)

print("Xử lý hoàn tất!")
print("Video kết quả:", TARGET_VIDEO_PATH)
print("File CSV:", csv_path)

# 12. Tạo báo cáo đơn giản
print("\n" + "="*50)
print("BÁO CÁO THỐNG KÊ GIAO THÔNG")
print("="*50)

for i, name in enumerate(region_names):
    total_txt = ", ".join([f"{CLASS_ID_TO_NAME[c]}={region_total_counts[i][c]}" for c in SELECTED_CLASS_IDS])
    print(f"Vùng {name}: {total_txt}")
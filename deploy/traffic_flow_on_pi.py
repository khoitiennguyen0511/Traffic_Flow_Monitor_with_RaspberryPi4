# ================== VERSION TỐI ƯU CHO RASPBERRY PI ==================
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from collections import defaultdict
import time
import os

# Giảm độ phân giải để tăng tốc độ xử lý
PI_OPTIMIZED = True

class RaspberryPiOptimizer:
    def __init__(self):
        self.frame_skip = 2  # Bỏ qua 1 frame, xử lý 1 frame
        self.processed_frames = 0
        self.resolution_scale = 0.6  # Giảm độ phân giải xuống 60%
        
    def should_process_frame(self):
        self.processed_frames += 1
        return self.processed_frames % self.frame_skip == 0
    
    def resize_frame(self, frame):
        if not PI_OPTIMIZED:
            return frame
        height, width = frame.shape[:2]
        new_width = int(width * self.resolution_scale)
        new_height = int(height * self.resolution_scale)
        return cv2.resize(frame, (new_width, new_height))

# Khởi tạo optimizer
pi_optimizer = RaspberryPiOptimizer()

# Sử dụng model nhẹ hơn cho Raspberry Pi
MODEL_PATH = "yolov8n.pt"  # Nano model thay vì best.pt

def setup_model():
    from ultralytics import YOLO
    import os
    
    # Kiểm tra nếu model chưa tồn tại thì tải về
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(MODEL_PATH)
    
    return model

# Khởi tạo model
print("Loading YOLO model...")
model = setup_model()

# Chỉ chọn các class chính để giảm tải xử lý
SELECTED_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Khởi tạo tracker với tham số tối ưu cho Pi
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.3,  # Tăng ngưỡng để giảm false positives
    lost_track_buffer=15,  # Giảm buffer để tiết kiệm bộ nhớ
    minimum_matching_threshold=0.7,
    frame_rate=15,  # Giảm FPS kỳ vọng
    minimum_consecutive_frames=2
)

# Cấu hình video
SOURCE_VIDEO_PATH = "traffic_video.mp4"  # Thay bằng video của bạn
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Tự động điều chỉnh độ phân giải dựa trên Pi performance
if PI_OPTIMIZED:
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
else:
    TARGET_WIDTH = video_info.width
    TARGET_HEIGHT = video_info.height

print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")

def P(px, py):
    return (int(px * TARGET_WIDTH), int(py * TARGET_HEIGHT))

def rect_frac(l, t, r, b):
    return [P(l, t), P(r, t), P(r, b), P(l, b)]

# Định nghĩa các vùng ROI đơn giản hóa
REGIONS_FRACTION = {
    "1": rect_frac(0.05, 0.3, 0.35, 0.8),
    "2": rect_frac(0.4, 0.1, 0.7, 0.3),
    "3": rect_frac(0.75, 0.3, 0.95, 0.8),
}

region_names = ["1", "2", "3"]
zones = []

for name in region_names:
    poly = np.array(REGIONS_FRACTION[name], dtype=np.int32)
    zone = sv.PolygonZone(polygon=poly)
    zones.append(zone)

# Biến đếm đơn giản hóa
region_counts = [defaultdict(int) for _ in zones]
region_present_ids = [set() for _ in zones]

# Annotators với độ phức tạp thấp
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.7)

def process_frame_simple(frame, frame_index):
    """Phiên bản xử lý frame tối ưu cho Raspberry Pi"""
    
    # Resize frame nếu cần
    if PI_OPTIMIZED and frame.shape[1] != TARGET_WIDTH:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Bỏ qua một số frame để giảm tải
    if not pi_optimizer.should_process_frame():
        return frame
    
    start_time = time.time()
    
    try:
        # Sử dụng inference với tham số tối ưu
        results = model.predict(
            frame,
            verbose=False,
            imgsz=320,  # Giảm kích thước inference
            conf=0.3,   # Ngưỡng confidence thấp hơn
            iou=0.5,    # Ngưỡng IoU
            half=False   # Không dùng half precision trên Pi
        )[0]
        
        # Chuyển đổi detections
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        
        # Tracking với độ phức tạp thấp
        if len(detections) > 0:
            detections = byte_tracker.update_with_detections(detections)
        
        # Xử lý từng vùng
        for zi, zone in enumerate(zones):
            inside_mask = zone.trigger(detections)
            
            for k, inside in enumerate(inside_mask):
                tid = detections.tracker_id[k] if detections.tracker_id is not None else None
                if tid is None:
                    continue
                    
                if inside and tid not in region_present_ids[zi]:
                    region_present_ids[zi].add(tid)
                    cid = detections.class_id[k]
                    region_counts[zi][cid] += 1
        
        # Vẽ kết quả (chỉ khi cần hiển thị)
        if PI_OPTIMIZED:
            # Vẽ bounding boxes đơn giản
            annotated_frame = box_annotator.annotate(frame.copy(), detections)
            
            # Vẽ các vùng ROI
            for zi, name in enumerate(region_names):
                poly = np.array(REGIONS_FRACTION[name], dtype=np.int32)
                cv2.polylines(annotated_frame, [poly], True, (0, 255, 0), 2)
                
                # Hiển thị số lượng
                total_count = sum(region_counts[zi].values())
                cx, cy = poly.mean(axis=0).astype(int)
                cv2.putText(annotated_frame, f"R{name}:{total_count}", 
                           (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame = annotated_frame
        
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Hiển thị FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    return frame

def run_realtime_analysis():
    """Chạy phân tích real-time từ webcam hoặc video"""
    
    # Sử dụng webcam mặc định
    cap = cv2.VideoCapture(0)
    
    # Cấu hình camera cho Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("Starting real-time vehicle counting...")
    print("Press 'q' to quit, 'r' to reset counters")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Xử lý frame
            processed_frame = process_frame_simple(frame, frame_count)
            
            # Hiển thị kết quả
            cv2.imshow('Vehicle Counting - Raspberry Pi', processed_frame)
            
            frame_count += 1
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset counters
                for zi in range(len(zones)):
                    region_counts[zi].clear()
                    region_present_ids[zi].clear()
                print("Counters reset!")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # In kết quả cuối cùng
        print("\nFinal Counts:")
        for zi, name in enumerate(region_names):
            total = sum(region_counts[zi].values())
            details = ", ".join([f"{CLASS_NAMES.get(cid, 'unknown')}:{count}" 
                               for cid, count in region_counts[zi].items()])
            print(f"Region {name}: Total={total} ({details})")

def run_video_analysis():
    """Phân tích video file"""
    if not os.path.exists(SOURCE_VIDEO_PATH):
        print(f"Video file not found: {SOURCE_VIDEO_PATH}")
        return
    
    output_path = "output_pi.mp4"
    
    # Process video với cài đặt tối ưu
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=output_path,
        callback=process_frame_simple
    )
    
    print(f"Analysis complete. Output saved to: {output_path}")

if __name__ == "__main__":
    print("Vehicle Counting System for Raspberry Pi")
    print("1. Real-time analysis (webcam)")
    print("2. Video file analysis")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        run_realtime_analysis()
    elif choice == "2":
        run_video_analysis()
    else:
        print("Invalid choice")
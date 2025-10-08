import os, cv2, time
import numpy as np
from collections import defaultdict, Counter
import supervision as sv
from ultralytics import YOLO

# ====== Config ======
MODEL_PATH = os.getenv("MODEL_PATH", "/home/pi/Traffic_Flow_Monitor_with_RaspberryPi4/best-int8.tflite")  # .tflite (khuyến nghị) hoặc .pt
SOURCE_PATH = os.getenv("SOURCE_PATH", "/home/pi/Traffic_Flow_Monitor_with_RaspberryPi4/vehicle_counting.mp4")
OUT_VIDEO  = os.getenv("OUT_VIDEO" , "/home/pi/Traffic_Flow_Monitor_with_RaspberryPi4/vehicle_counting_4regions_pi.mp4")

SELECTED_CLASS_NAMES = ["motorbike", "car", "bus", "truck"]
CONF_THRES = float(os.getenv("CONF", "0.35"))
IOU_THRES  = float(os.getenv("IOU", "0.5"))
IMG_SIZE   = int(os.getenv("IMGSZ", "384"))    # 320–480 hợp lý cho Pi
TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))
DRAW_TRACE = os.getenv("TRACE", "0") == "1"    # bật vẽ quỹ đạo nếu cần

FLOW_INTERVAL = int(os.getenv("FLOW_SEC", "60"))  # vẫn tính lưu lượng để overlay, không ghi CSV

# ====== Load model ======
print(f"[INFO] Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
CLASS_NAMES_DICT = model.model.names  # id -> name

# map class names -> ids có trong model
name2id = {v:k for k, v in CLASS_NAMES_DICT.items()}
SELECTED_CLASS_IDS = [name2id[n] for n in SELECTED_CLASS_NAMES if n in name2id]
CLASS_ID_TO_NAME = {cid: CLASS_NAMES_DICT[cid] for cid in SELECTED_CLASS_IDS}

# ====== Video IO ======
cap = cv2.VideoCapture(SOURCE_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {SOURCE_PATH}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SRC_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
FPS = max(1, int(round(SRC_FPS)))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Nếu ghi chậm: dùng MJPG + .avi
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, min(TARGET_FPS, FPS), (W, H))
print(f"[INFO] Video {W}x{H}, src_fps={SRC_FPS:.2f}, out_fps={min(TARGET_FPS,FPS)}")

# ====== Regions (tỉ lệ -> pixel) ======
def P(px, py): return (int(px * W), int(py * H))
def rect_frac(l, t, r, b): return [P(l, t), P(r, t), P(r, b), P(l, b)]

REGIONS_FRACTION = {
    "1": rect_frac(0.01, 0.28, 0.22, 0.90),
    "2": rect_frac(0.30, 0.01, 0.78, 0.22),
    "3": rect_frac(0.80, 0.22, 0.99, 0.85),
    "4": rect_frac(0.23, 0.88, 0.72, 0.99),
}
region_names = ["1","2","3","4"]

REGION_COLORS = {"normal": sv.Color(255,0,0), "active": sv.Color(0,255,0)}

zones, zone_annots = [], []
class CustomPolygonZoneAnnotator:
    def __init__(self, zone, color, thickness=3):
        self.zone = zone; self.color = color; self.thickness = thickness
    def annotate(self, scene, color_override=None):
        color = color_override if color_override else self.color
        cv2.polylines(scene, [self.zone.polygon], True, color.as_bgr(), self.thickness)
        return scene

for name in region_names:
    poly = np.array(REGIONS_FRACTION[name], dtype=np.int32)
    zone = sv.PolygonZone(polygon=poly)
    zones.append(zone)
    zone_annots.append(CustomPolygonZoneAnnotator(zone=zone, color=REGION_COLORS["normal"]))

# ====== ByteTrack ======
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=FPS,
    minimum_consecutive_frames=2
)
byte_tracker.reset()

# ====== Counters (không ghi CSV) ======
region_total_counts = [defaultdict(int) for _ in zones]
region_present_ids  = [set() for _ in zones]
region_active_frames = [0 for _ in zones]
ACTIVE_DURATION = 30

current_flow_counts = [defaultdict(int) for _ in zones]
flow_start_time = 0.0

# ====== Vẽ ======
trace_annotator = sv.TraceAnnotator(thickness=3, trace_length=40)
box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)

# ====== Ổn định nhãn theo tracker_id ======
SMOOTH_WIN = 12
LOCK_AFTER = 5
id_class_hist  = defaultdict(lambda: [])
id_locked_class = {}

def stabilize_class(tid, cid):
    if tid in id_locked_class:
        return id_locked_class[tid]
    hist = id_class_hist[tid]
    hist.append(int(cid))
    if len(hist) > SMOOTH_WIN:
        hist.pop(0)
    if len(hist) >= LOCK_AFTER and all(x == cid for x in hist[-LOCK_AFTER:]):
        id_locked_class[tid] = int(cid)
        return id_locked_class[tid]
    # majority vote
    return int(Counter(hist).most_common(1)[0][0])

# ====== Loop ======
index = 0
t0 = time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    index += 1
    current_time = index / FPS

    # Inference (TFLite/CPU)
    results = model(frame, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMG_SIZE, verbose=False)[0]
    det = sv.Detections.from_ultralytics(results)
    if len(det) == 0:
        writer.write(frame); continue

    det = det[np.isin(det.class_id, SELECTED_CLASS_IDS)]
    det = byte_tracker.update_with_detections(det)

    # ổn định class theo tracker_id
    for k in range(len(det)):
        tid = det.tracker_id[k]
        if tid is None: continue
        det.class_id[k] = stabilize_class(int(tid), int(det.class_id[k]))

    # Vẽ bbox/label (trace tùy chọn để tiết kiệm CPU)
    annotated = frame
    if DRAW_TRACE:
        annotated = trace_annotator.annotate(annotated, det)

    labels = [f"#{tid} {CLASS_NAMES_DICT[int(c)]} {float(p):.2f}"
              for p, c, tid in zip(det.confidence, det.class_id, det.tracker_id)]
    annotated = box_annotator.annotate(annotated, det)
    annotated = label_annotator.annotate(annotated, det, labels=labels)

    # Đếm theo 4 vùng + overlay (không ghi CSV)
    for zi, zone in enumerate(zones):
        inside_mask = zone.trigger(det)
        new_entry = False
        for k, inside in enumerate(inside_mask):
            tid = det.tracker_id[k]
            if tid is None: continue
            if inside and tid not in region_present_ids[zi]:
                region_present_ids[zi].add(tid)
                cid = int(det.class_id[k])
                region_total_counts[zi][cid] += 1
                current_flow_counts[zi][cid] += 1
                new_entry = True
                region_active_frames[zi] = ACTIVE_DURATION

        # cửa sổ lưu lượng (chỉ để hiển thị)
        if current_time - flow_start_time >= FLOW_INTERVAL:
            current_flow_counts = [defaultdict(int) for _ in zones]
            flow_start_time = current_time

        # màu vùng + text
        if region_active_frames[zi] > 0:
            region_active_frames[zi] -= 1
        cur_color = REGION_COLORS["active"] if region_active_frames[zi] > 0 else REGION_COLORS["normal"]
        annotated = zone_annots[zi].annotate(annotated, color_override=cur_color)

        poly = np.array(REGIONS_FRACTION[region_names[zi]], dtype=np.float32)
        cx, cy = poly.mean(axis=0).astype(int)
        occ = int(np.sum(inside_mask))
        cur_flow_per_min = sum(current_flow_counts[zi].values()) * (60 / FLOW_INTERVAL)
        text_color = (0,255,0) if region_active_frames[zi] > 0 else (0,0,255)

        cv2.putText(annotated, f"Region {region_names[zi]}  Occ:{occ}", (cx-170, cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        summary = " | ".join([f"{CLASS_ID_TO_NAME[c]}:{region_total_counts[zi][c]}" for c in SELECTED_CLASS_IDS])
        cv2.putText(annotated, summary, (cx-220, cy+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Flow: {cur_flow_per_min:.1f}/min", (cx-170, cy+90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

    writer.write(annotated)

cap.release()
writer.release()
print(f"[DONE] Saved video to: {OUT_VIDEO}")
print(f"[INFO] Approx FPS: {index / (time.time() - t0):.2f}")
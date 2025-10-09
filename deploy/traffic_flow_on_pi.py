import os
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
import onnxruntime as ort

# ================= CONFIGURATION =================
MODEL_PATH = "best.onnx"
SOURCE_VIDEO_PATH = "vehicle_counting.mp4"
TARGET_VIDEO_PATH = "output_counting.mp4"

# ================= CHECK MODEL =================
print("Checking ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# Get model input info
input_info = session.get_inputs()[0]
input_name = input_info.name
input_shape = input_info.shape
print(f"Model input shape: {input_shape}")

# Determine required model size
# Usually: [1, 3, height, width]
MODEL_HEIGHT = input_shape[2]  # 384
MODEL_WIDTH = input_shape[3]   # 384

print(f"Model requires size: {MODEL_WIDTH}x{MODEL_HEIGHT}")

# Class names (adjust according to your model)
CLASS_NAMES_DICT = {0: "motorbike", 1: "car", 2: "bus", 3: "truck"}
SELECTED_CLASS_IDS = [0, 1, 2, 3]

# ================= INITIALIZE VIDEO =================
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))  # Fixed: Use FPS property, not FRAME_COUNT

print(f"Video: {W}x{H} @ {FPS}fps")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, FPS, (W, H))

# ================= INITIALIZE BYTETRACK =================
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=FPS
)

# ================= DEFINE ROI REGIONS =================
def scale_coords(px, py):
    return (int(px * W), int(py * H))

def create_region(l, t, r, b):
    return [scale_coords(l, t), scale_coords(r, t), 
            scale_coords(r, b), scale_coords(l, b)]

REGIONS = {
    "1": create_region(0.01, 0.28, 0.22, 0.9),
    "2": create_region(0.3, 0.01, 0.78, 0.22),
    "3": create_region(0.8, 0.22, 0.99, 0.85),
    "4": create_region(0.23, 0.88, 0.72, 0.99),
}

# Create zones
zones = []
for name, polygon in REGIONS.items():
    zone = sv.PolygonZone(polygon=np.array(polygon))
    zones.append(zone)

# ================= FIXED INFERENCE FUNCTION =================
def inference_onnx(frame):
    try:
        # Preprocess - resize to model required size
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (MODEL_WIDTH, MODEL_HEIGHT))
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        # Inference
        outputs = session.run(None, {input_name: input_img})
        
        # Post-process - assuming output is YOLO format
        predictions = np.squeeze(outputs[0]).T
        
        # Get number of classes from output shape
        num_classes = predictions.shape[1] - 4
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:4+num_classes]
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence
        valid_detections = class_scores > 0.25
        boxes = boxes[valid_detections]
        class_scores = class_scores[valid_detections]
        class_ids = class_ids[valid_detections]
        
        if len(boxes) == 0:
            return sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x_center - width / 2) * (W / MODEL_WIDTH)
        y1 = (y_center - height / 2) * (H / MODEL_HEIGHT)
        x2 = (x_center + width / 2) * (W / MODEL_WIDTH)
        y2 = (y_center + height / 2) * (H / MODEL_HEIGHT)
        
        # Create detections
        detections = sv.Detections(
            xyxy=np.column_stack([x1, y1, x2, y2]),
            confidence=class_scores,
            class_id=class_ids.astype(int)
        )
        
        # Filter desired classes
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        
        return detections
        
    except Exception as e:
        print(f"Inference error: {e}")
        return sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))

# ================= ANNOTATORS =================
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=15)

# ================= COUNTING VARIABLES =================
region_counts = [defaultdict(int) for _ in range(4)]
region_active_ids = [set() for _ in range(4)]

# ================= PROCESS VIDEO =================
print("Starting video processing...")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    detections = inference_onnx(frame)
    
    if len(detections) > 0:
        # Tracking
        detections = byte_tracker.update_with_detections(detections)
        
        # Annotate
        annotated_frame = trace_annotator.annotate(frame.copy(), detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        
        # Labels
        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            if tracker_id is not None:
                labels.append(f"#{tracker_id} {CLASS_NAMES_DICT.get(class_id, 'unknown')}")
        
        if labels:
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
    else:
        annotated_frame = frame.copy()
    
    # Process each region
    for i, zone in enumerate(zones):
        # Trigger zone
        trigger_mask = zone.trigger(detections) if len(detections) > 0 else np.array([])
        current_occupancy = np.sum(trigger_mask) if len(trigger_mask) > 0 else 0
        
        # Count new vehicles entering region
        if len(detections) > 0:
            for j, (inside, tracker_id) in enumerate(zip(trigger_mask, detections.tracker_id)):
                if inside and tracker_id is not None and tracker_id not in region_active_ids[i]:
                    region_active_ids[i].add(tracker_id)
                    class_id = detections.class_id[j]
                    region_counts[i][class_id] += 1
        
        # Draw zone
        region_poly = np.array(REGIONS[str(i+1)])
        cv2.polylines(annotated_frame, [region_poly], True, (0, 255, 0), 2)
        
        # Display information
        center_x = int(np.mean(region_poly[:, 0]))
        center_y = int(np.mean(region_poly[:, 1]))
        
        # Total count
        count_text = " ".join([f"{CLASS_NAMES_DICT[cid]}:{region_counts[i][cid]}" 
                              for cid in SELECTED_CLASS_IDS if cid in CLASS_NAMES_DICT])
        
        # Draw information
        cv2.putText(annotated_frame, f"Region {i+1}: {current_occupancy}", 
                   (center_x - 180, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 0), 4)
        cv2.putText(annotated_frame, count_text, 
                   (center_x - 200, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 3)
    
    # Display frame
    cv2.imshow('Vehicle Counting', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Write video
    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

# ================= CLEANUP =================
cap.release()
out.release()
cv2.destroyAllWindows()

# ================= REPORT =================
print("\n" + "="*50)
print("VEHICLE COUNTING RESULTS")
print("="*50)

total_vehicles = 0
for i in range(4):
    region_total = sum(region_counts[i].values())
    total_vehicles += region_total
    details = " | ".join([f"{CLASS_NAMES_DICT[cid]}: {region_counts[i][cid]}" 
                         for cid in SELECTED_CLASS_IDS if cid in CLASS_NAMES_DICT])
    print(f"Region {i+1}: {region_total} vehicles - {details}")

print(f"\nTOTAL: {total_vehicles} vehicles")
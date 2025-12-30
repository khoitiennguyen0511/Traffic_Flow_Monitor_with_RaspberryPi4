import cv2
import numpy as np
import supervision as sv
from collections import defaultdict
import onnxruntime as ort
import paho.mqtt.client as mqtt
import json

# ================= CONFIGURATION =================
MODEL_PATH = "best.onnx"

# MQTT
MQTT_BROKER = "172.20.10.5"
MQTT_PORT = 1883
TOPIC_GREEN_TIME_CMD = "he_thong_giam_sat_luu_luong/green_time_cmd"
TOPIC_MANUAL_CMD = "he_thong_giam_sat_luu_luong/control"
TOPIC_VEHICLE_COUNT = "he_thong_giam_sat_luu_luong/vehicle_count"

# Classes
CLASS_NAMES_DICT = {0: "motorbike", 1: "car", 2: "bus", 3: "truck"}
SELECTED_CLASS_IDS = [0, 1, 2, 3]

# ================= INITIALIZE MQTT =================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(TOPIC_MANUAL_CMD)
        client.subscribe(TOPIC_GREEN_TIME_CMD)
    else:
        print(f"Failed to connect MQTT, rc={rc}")

def on_message(client, userdata, msg):
    message = msg.payload.decode()
    print(f"Received MQTT message from {msg.topic}: {message}")
    if msg.topic == TOPIC_MANUAL_CMD:
        print(f"Traffic mode change request: {message}")
    elif msg.topic == TOPIC_GREEN_TIME_CMD:
        try:
            green_time = int(message)
            print(f"Update green_time to {green_time} seconds")
        except:
            pass

mqtt_client = mqtt.Client(client_id="RPi_TrafficCounter")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
    mqtt_client.loop_start()
except Exception as e:
    print(f"Could not connect to MQTT Broker: {e}")

# ================= INITIALIZE WEBCAM & MODEL & TRACKER =================
cap = cv2.VideoCapture(0)  # Webcam máº·c Ä‘á»‹nh
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
if FPS == 0:
    FPS = 30  # Má»™t sá»‘ webcam tráº£ FPS=0
print(f"Camera resolution: {W}x{H}, FPS={FPS}")

# Náº¿u muá»‘n lÆ°u video, uncomment:
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("webcam_output.mp4", fourcc, FPS, (W, H))

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_info = session.get_inputs()[0]
input_name = input_info.name
MODEL_HEIGHT = input_info.shape[2]
MODEL_WIDTH = input_info.shape[3]
print(f"Model input size: {MODEL_WIDTH}x{MODEL_HEIGHT}")

# Tracker
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=FPS
)

# ================= HELPER FUNCTIONS =================
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

zones = [sv.PolygonZone(polygon=np.array(polygon)) for polygon in REGIONS.values()]
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=15)
region_counts = [defaultdict(int) for _ in range(4)]
region_active_ids = [set() for _ in range(4)]

def inference_onnx(frame):
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (MODEL_WIDTH, MODEL_HEIGHT))
    input_img = input_img.transpose(2,0,1)[np.newaxis,...].astype(np.float32)/255.0
    outputs = session.run(None, {input_name: input_img})
    predictions = np.squeeze(outputs[0]).T
    if predictions.shape[0] == 0:
        return sv.Detections(xyxy=np.array([]), confidence=np.array([]), class_id=np.array([]))
    num_classes = predictions.shape[1]-4
    boxes = predictions[:, :4]
    scores = predictions[:, 4:4+num_classes]
    class_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    valid = class_scores > 0.25
    boxes, class_scores, class_ids = boxes[valid], class_scores[valid], class_ids[valid]
    x_center, y_center, width, height = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = (x_center - width/2)*(W/MODEL_WIDTH)
    y1 = (y_center - height/2)*(H/MODEL_HEIGHT)
    x2 = (x_center + width/2)*(W/MODEL_WIDTH)
    y2 = (y_center + height/2)*(H/MODEL_HEIGHT)
    detections = sv.Detections(
        xyxy=np.column_stack([x1,y1,x2,y2]),
        confidence=class_scores,
        class_id=class_ids.astype(int)
    )
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    return detections

def publish_counts(region_counts):
    payload = {}
    for i, region in enumerate(region_counts):
        cleaned_region_data = {CLASS_NAMES_DICT[int(cid)]: int(count) for cid, count in region.items()}
        payload[f"region_{i+1}"] = cleaned_region_data
    try:
        mqtt_client.publish(TOPIC_VEHICLE_COUNT, json.dumps(payload))
    except TypeError as e:
        print(f"MQTT Publish Error: {e}")

# ================= MAIN LOOP =================
frame_count = 0
print("Starting webcam processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot get frame from webcam")
        break

    detections = inference_onnx(frame)

    if len(detections) > 0:
        detections = byte_tracker.update_with_detections(detections)
        annotated_frame = trace_annotator.annotate(frame.copy(), detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        labels = [f"#{tid} {CLASS_NAMES_DICT.get(cid,'unknown')}" 
                  for cid, tid in zip(detections.class_id, detections.tracker_id) if tid is not None]
        if labels:
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
    else:
        annotated_frame = frame.copy()

    # Process zones
    for i, zone in enumerate(zones):
        trigger_mask = zone.trigger(detections) if len(detections)>0 else np.array([])
        if len(detections)>0:
            for j, (inside, tid) in enumerate(zip(trigger_mask, detections.tracker_id)):
                if inside and tid is not None and tid not in region_active_ids[i]:
                    region_active_ids[i].add(tid)
                    cid = detections.class_id[j]
                    region_counts[i][cid] += 1
        region_poly = np.array(REGIONS[str(i+1)])
        cv2.polylines(annotated_frame,[region_poly],True,(0,255,0),2)
        cx, cy = int(np.mean(region_poly[:,0])), int(np.mean(region_poly[:,1]))
        count_text = " ".join([f"{CLASS_NAMES_DICT[int(cid)]}:{int(count)}" 
                               for cid, count in region_counts[i].items() if int(cid) in CLASS_NAMES_DICT])
        cv2.putText(annotated_frame,f"Region {i+1}",(cx-60,cy-25),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(annotated_frame,count_text,(cx-60,cy+25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Vehicle Counting", annotated_frame)

    # Náº¿u muá»‘n lÆ°u video, uncomment:
    # out.write(annotated_frame)

    frame_count += 1
    if frame_count % FPS == 0:
        publish_counts(region_counts)
        print(f"Published counts to MQTT at frame {frame_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()  # Náº¿u lÆ°u video
cv2.destroyAllWindows()
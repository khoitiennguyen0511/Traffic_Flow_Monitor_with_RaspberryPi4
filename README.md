# Vehicle Detection and Tracking System

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=white)](https://ultralytics.com/)
[![ByteTRACK](https://img.shields.io/badge/ByteTRACK-FF6B6B?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge)](https://onnx.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Hệ thống phát hiện và theo dõi phương tiện giao thông thời gian thực sử dụng YOLOv8 và ByteTRACK, hỗ trợ triển khai trên nhiều nền tảng với các format model tối ưu.

## Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năn nổi bật](#-tính-năng-nổi-bật)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Mô hình](#-mô-hình)
- [Kết quả](#-kết-quả)
- [Đóng góp](#-đóng-góp)
- [License](#-license)
- [Liên hệ](#-liên-hệ)

## Giới thiệu

Dự án này cung cấp giải pháp cho bài toán phát hiện và theo dõi phương tiện giao thông trong video thời gian thực. Hệ thống kết hợp YOLOv8n cho việc phát hiện đối tượng chính xác và ByteTRACK cho việc theo dõi đa đối tượng ổn định.

## Tính năng nổi bật

- **Phát hiện phương tiện chính xác** với YOLOv8n
- **Theo dõi đa đối tượng** với ByteTRACK
- **Tối ưu hóa hiệu suất** với ONNX
- **Xử lý dữ liệu thông minh** với Jupyter Notebook
- **Triển khai** trên thiết bị edge

## Cấu trúc dự án
vehicle-detection-tracking/
├── 📁 weights/ # Thư mục chứa mô hình
│ ├── best.onnx # Mô hình ONNX để inference nhanh
│ └── best_int8.tflite # Mô hình TFLite quantized cho thiết bị edge
├── 📁 runs/detect/ # Kết quả inference và tracking
├── 📄 Data_Processing.ipynb # Tiền xử lý và phân tích dữ liệu
├── 📄 Vehicle_Detection_YOLOv8.ipynb # Phát hiện phương tiện với YOLOv8
├── 📄 YOLOv8n_ByteTRACK_Tracking.ipynb # Theo dõi với ByteTRACK
├── 📄 .gitignore # Cấu hình loại trừ file lớn
└── 📄 README.md # Tài liệu dự án

text

## ⚡ Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.0+ (cho GPU)
- RAM 8GB+
- Storage 2GB+
- Raspberry pi 4 (4GB hoặc 8GB)

### Train trên Google Colab
1. Xử lý dữ liệu
Mở notebook Data_Processing.ipynb và chạy các cell theo thứ tự:

2. Phát hiện phương tiện cơ bản
Mở notebook Vehicle_Detection_YOLOv8.ipynb và chạy các cell theo thứ tự: 

3. Theo dõi với ByteTRACK
Mở notebook YOLOv8n_ByteTRACK_Tracking.ipynb và chạy các cell theo thứ tự: 
### Cài đặt dependencies trên Raspberry pi 4

```bash
# Clone repository
git clone https://github.com/khoitiennguyen0511/vehicle-detection-tracking.git
cd vehicle-detection-tracking

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies cơ bản
pip install ultralytics torch torchvision

# Cài đặt dependencies cho inference
pip install onnxruntime tensorflow

# Cài đặt Jupyter Notebook
pip install jupyter notebook

# Cài đặt thư viện hỗ trợ
pip install opencv-python pandas numpy matplotlib
Cài đặt nhanh (đầy đủ)
bash
pip install -r requirements.txt
Lưu ý: Tạo file requirements.txt nếu cần

🚀 Sử dụng
1. Phát hiện phương tiện cơ bản
Mở notebook Vehicle_Detection_YOLOv8.ipynb và chạy các cell theo thứ tự:

python
# Load model YOLOv8
from ultralytics import YOLO
model = YOLO('weights/best.onnx')

# Detection trên ảnh
results = model('path/to/image.jpg', save=True, conf=0.25)

# Detection trên video
results = model('path/to/video.mp4', save=True, conf=0.25)
2. Theo dõi với ByteTRACK
Mở notebook YOLOv8n_ByteTRACK_Tracking.ipynb:

python
# Tracking với ByteTRACK
results = model.track('path/to/video.mp4', 
                     tracker='bytetrack.yaml', 
                     conf=0.25, 
                     iou=0.7,
                     save=True)
3. Xử lý dữ liệu
Mở notebook Data_Processing.ipynb để:

Phân tích dataset

Tiền xử lý dữ liệu

Visualize kết quả

4. Sử dụng model TFLite
python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="weights/best_int8.tflite")
interpreter.allocate_tensors()

# Lấy input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
📊 Mô hình
Thông tin mô hình
Model	Format	Precision	Use Case	Inference Speed
YOLOv8n	ONNX	FP32	Server/High-performance	⭐⭐⭐⭐
YOLOv8n	TFLite	INT8	Mobile/IoT Devices	⭐⭐⭐
Classes được hỗ trợ
🚗 Car (ô tô)

🚙 SUV

🚐 Van

🚑 Ambulance

🚒 Fire truck

🚛 Truck (xe tải)

🚌 Bus (xe buýt)

🛵 Motorcycle (xe máy)

Performance
mAP@0.5: 0.85+

Inference Speed: 15-30 FPS (trên RTX 3060)

Accuracy: 90%+ trên dataset validation

📈 Kết quả
Output Structure
text
runs/
└── detect/
    ├── predict/              # Kết quả detection
    │   ├── image1.jpg
    │   ├── video1.mp4
    │   └── labels/           # File labels
    └── track/               # Kết quả tracking
        ├── video1_track.mp4
        └── tracks.txt       # Dữ liệu tracking
Visualizations
✅ Bounding boxes với confidence scores

✅ ID tracking duy nhất cho mỗi vehicle

✅ Motion trails cho tracking visualization

✅ Export results dưới dạng video và images

⚙️ Tùy chỉnh
Điều chỉnh tham số
python
# Cấu hình tracking
tracking_config = {
    'tracker_type': 'bytetrack',
    'conf': 0.25,           # Ngưỡng confidence
    'iou': 0.7,             # Ngưỡng IoU
    'classes': [2, 3, 5, 7], # Classes phương tiện
    'persist': True,        # Duy trì ID across frames
    'show': True           # Hiển thị kết quả real-time
}
Custom Classes
python
# Chỉ detect car và motorcycle
results = model('input.jpg', classes=[2, 3])
🐛 Xử lý lỗi thường gặp
Lỗi GPU/CUDA
bash
# Kiểm tra CUDA
nvidia-smi

# Cài đặt torch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Lỗi memory
python
# Giảm batch size
results = model('input.jpg', batch_size=4)
Lỗi model loading
python
# Sử dụng model mặc định nếu custom model lỗi
model = YOLO('yolov8n.pt')
🤝 Đóng góp
Đóng góp luôn được chào đón! Vui lòng làm theo các bước:

Fork repository

Tạo feature branch

bash
git checkout -b feature/AmazingFeature
Commit changes

bash
git commit -m 'Add some AmazingFeature'
Push to branch

bash
git push origin feature/AmazingFeature
Open Pull Request

Guidelines
Tuân thủ PEP 8 coding style

Thêm comments cho code mới

Cập nhật documentation

Test kỹ trước khi commit

📄 License
Dự án được phân phối dưới MIT License. Xem file LICENSE để biết thêm chi tiết.

👤 Liên hệ
Khoi Tien Nguyen

GitHub: @khoitiennguyen0511

Email: [your-email@domain.com]

🙏 Acknowledgments
Ultralytics cho YOLOv8

ByteTRACK cho multi-object tracking

Cộng đồng AI/ML Việt Nam

⭐ Nếu bạn thấy dự án hữu ích, đừng quên cho repository một star!




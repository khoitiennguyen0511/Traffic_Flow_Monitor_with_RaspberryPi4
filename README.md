# Traffic Flow Monitor using Yolo8 + ByteTRACK on Raspberry pi 4

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=white)](https://ultralytics.com/)
[![ByteTRACK](https://img.shields.io/badge/ByteTRACK-FF6B6B?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge)](https://onnx.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Hệ thống giám sát lưu lượng giao thông thời gian thực kết hợp **YOLOv8** (phát hiện) và **ByteTRACK** (theo dõi đa mục tiêu), tối ưu cho **Raspberry Pi 4** và thiết bị edge.

**Demo video:** [YouTube](https://www.youtube.com/watch?v=68LdN0nzT2w)

<p align="center"> <img src="https://github.com/khoitiennguyen0511/Traffic_Flow_Monitor_with_RaspberryPi4/raw/main/assets/traffic-demo.gif" alt="Traffic Flow Monitor Demo — YOLOv8 + ByteTrack trên Raspberry Pi 4" width="800"> <br> <em>Demo thực tế - Giám sát lưu lượng giao thông trên Raspberry pi 4</em> </p>

---

## Tổng quan
Pipeline hoàn chỉnh để **phát hiện** và **theo dõi** phương tiện trong video thời gian thực, đồng thời **đếm lượt vào/ra theo vùng** (ROI/Polygon) nhằm ước lượng lưu lượng. Trọng tâm là triển khai nhẹ, ổn định trên **Raspberry Pi 4 (64-bit)** bằng **ONNX Runtime**.

## Tính năng
- **Phát hiện thời gian thực**: Nhận diện nhiều lớp phương tiện (ví dụ: `bus`, `motorbike`, `car`, `truck`, …).
- **Theo dõi đa mục tiêu**: Gán ID ổn định giữa các khung hình bằng **ByteTRACK**.
- **Đếm theo vùng/ROI**: Tổng hợp lượt vào/ra theo từng vùng, log & overlay trực tiếp.
- **Tối ưu cho edge**: Mô hình **ONNX** + thiết lập inference hợp lý trên Raspberry Pi.
- **Đa nguồn vào**: Hỗ trợ webcam và file video.

---

## Cấu trúc dự án
```bash
Traffic_Flow_Monitor_with_RaspberryPi4/
├── weights/
│   ├── best.onnx           # Model ONNX tối ưu cho Pi
│   └── best.pt             # Checkpoint YOLOv8 (tùy chọn)
├── runs/
│   └── detect/             # Kết quả inference/track
├── deploy/
│   └── traffic_flow_on_pi.py
├── Data_Processing.ipynb
├── Vehicle_Detection_YOLOv8.ipynb
├── YOLOv8n_ByteTRACK_Tracking.ipynb
├── .gitignore
└── README.md
```

## Chạy trên Google Colab
1. Xử lý dữ liệu: [Data_Processing.ipynb](https://colab.research.google.com/drive/1FKN6ic0ZNOxkFsI2u88UP7lIdtP9fKf7)

3. Phát hiện với YOLOv8n: [Vehicle_Detection_YOLOv8.ipynb](https://colab.research.google.com/drive/1epMKc-LLfKyHEd-rQiLBMnck3pUON62s)

4. Theo dõi với ByteTRACK: [YOLOv8n_ByteTRACK_Tracking.ipynb](https://colab.research.google.com/drive/1FKN6ic0ZNOxkFsI2u88UP7lIdtP9fKf7)

## Cài đặt

### Yêu cầu hệ thống

- Raspberry Pi 4 (4GB/8GB RAM) và thẻ SD card
- Python 3.8+
- Raspberry Pi OS (64-bit)

### Cài đặt dependencies trên Raspberry pi 4

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv python3-opencv libopencv-dev ffmpeg

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy==2.2.6 opencv-python 4.12.0.88 supervision==0.26.1 onnxruntime==1.23.1 ultralytics 8.3.207 torch 2.8.0

# Git clone this repo
git clone https://github.com/khoitiennguyen0511/Traffic_Flow_Monitor_with_RaspberryPi4.git

# Create a new folder
mkdir my_project
cd my_project
```
## Sao chép các files vào folder my_project
- **traffic_flow_on_pi.py** trong folder deploy
- **best.onnx** trong folder weights
- **vehicle_counting.mp4**

## Chạy chương trình
```bash
python3 traffic_flow_on_pi.py
```

## Liên hệ
GitHub: @khoitiennguyen0511
Email: [khoitiennguyen2004l@gmail.com]
Linkln: [czxc]

⭐ Nếu bạn thấy dự án hữu ích, đừng quên cho repository một star!


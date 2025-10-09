# Traffic Flow Monitor using Yolo8 + ByteTRACK on Raspberry pi 4

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=white)](https://ultralytics.com/)
[![ByteTRACK](https://img.shields.io/badge/ByteTRACK-FF6B6B?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge)](https://onnx.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Hệ thống giám sát lưu lượng giao thông thời gian thực kết hợp YOLOv8 phát hiện đối tượng với ByteTRACK theo dõi đa mục tiêu, tối ưu hóa cho triển khai trên Raspberry Pi 4 và các thiết bị edge.

<p align="center"> <img src="https://github.com/khoitiennguyen0511/Traffic_Flow_Monitor_with_RaspberryPi4/raw/main/assets/traffic-demo.gif" alt="Traffic Flow Monitor Demo — YOLOv8 + ByteTrack trên Raspberry Pi 4" width="800"> <br> <em>Demo thực tế - Giám sát lưu lượng giao thông trên Raspberry pi 4</em> </p>

**Demo video:** [YouTube](https://www.youtube.com/watch?v=68LdN0nzT2w)

---

## Giới thiệu

Dự án này cung cấp giải pháp cho bài toán phát hiện và theo dõi phương tiện giao thông trong video thời gian thực. Hệ thống kết hợp YOLOv8n cho việc phát hiện đối tượng chính xác và ByteTRACK cho việc theo dõi đa đối tượng ổn định.

## Tính năng nổi bật

- **Phát hiện phương tiện thời gian thực**: Nhận diện chính xác nhiều loại phương tiện như _bus, motorbike, car, truck_
_
- **Theo dõi đa mục tiêu**: Gán ID nhất quán xuyên suốt các frame sử dụng ByteTRACK

- **Tối ưu hóa edge**: Tối ưu runtime ONNX cho triển khai Raspberry Pi

- **Đa nguồn đầu vào**: Hỗ trợ webcam và file video

- **Kết quả**: Hiển thị kết quả lượng xe vào/ra theo từng vùng

## Cấu trúc dự án
```bash
vehicle-detection-tracking/
├── weights/ 
│ ├── best.onnx
│ └── best.pt 
├── runs/ 
│ └── detect/   # Kết quả inference
├── Data_Processing.ipynb
├── Vehicle_Detection_YOLOv8.ipynb
├── YOLOv8n_ByteTRACK_Tracking.ipynb
├── .gitignore
└── README.md
```

## Chạy trên Google Colab
1. Xử lý dữ liệu: [Data_Processing.ipynb](https://colab.research.google.com/drive/1FKN6ic0ZNOxkFsI2u88UP7lIdtP9fKf7)

3. Phát hiện phương tiện: [Vehicle_Detection_YOLOv8.ipynb](https://colab.research.google.com/drive/1epMKc-LLfKyHEd-rQiLBMnck3pUON62s)

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

# 📦 Smart Inventory Management System using YOLO

This project implements a **Smart Inventory Management System** that uses **YOLO (You Only Look Once) object detection** to track and manage products in real-time.  
It enables automatic product recognition, inventory updates, and predictive analytics to optimize warehouse and store management.

---

## 📘 Project Overview
The system captures images or video streams from a warehouse/store environment and detects individual products using YOLO.  
Key functionalities include:  
- **Object detection** for identifying products on shelves  
- **Real-time inventory updates** based on detected objects  
- **Predictive analytics** to forecast inventory demand  
- Integration of **voice commands** for live interaction with the system  
- Data storage and visualization for inventory tracking

---

## 🛠️ Tech Stack
- **Languages & Libraries**: Python, OpenCV, NumPy, Pandas, PyTorch  
- **Object Detection**: YOLOv8  
- **Microcontrollers**: ESP32 (for live image collection and IoT integration)  
- **Cloud & ML**: AWS SageMaker (for model training), S3 (storage)  
- **Visualization & Analytics**: Power BI / Matplotlib for dashboard creation  
- **Other Tools**: pyttsx3 for voice output  

---

## ⚙️ Workflow

1. **Data Collection**
   - Images collected from store/warehouse using cameras or ESP32 devices  
   - Preprocessing images for training YOLO  

2. **Model Training**
   - YOLOv8 trained on labeled dataset of products  
   - Validation and testing performed to ensure accuracy  

3. **Object Detection & Inventory Updates**
   - Real-time object detection using YOLO  
   - Detected products logged in inventory database  
   - Automated updates for stock levels  

4. **Predictive Analytics**
   - Use historical inventory data to forecast product demand  
   - Generate actionable insights for restocking and supply chain optimization  

---


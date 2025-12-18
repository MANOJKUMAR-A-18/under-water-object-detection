#!/usr/bin/env python3
"""
Quick script to check available trained models
"""

import os

def check_models():
    print("🔍 Checking for trained models...")
    
    # Check YOLOv8n model
    yolov8n_path = "../yolov8n/runs/detect_train/weights/best.pt"
    if os.path.exists(yolov8n_path):
        size = os.path.getsize(yolov8n_path) / (1024*1024)  # MB
        print(f"✅ YOLOv8n model found: {yolov8n_path} ({size:.1f} MB)")
    else:
        print(f"❌ YOLOv8n model not found: {yolov8n_path}")
    
    # Check YOLOv8s model
    yolov8s_path = "../yolov8s/runs/detect_train/weights/best.pt"
    if os.path.exists(yolov8s_path):
        size = os.path.getsize(yolov8s_path) / (1024*1024)  # MB
        print(f"✅ YOLOv8s model found: {yolov8s_path} ({size:.1f} MB)")
    else:
        print(f"❌ YOLOv8s model not found: {yolov8s_path}")
    
    # Check if we can run the app
    if os.path.exists(yolov8n_path) or os.path.exists(yolov8s_path):
        print("\n🎉 You have trained models! You can run the app:")
        print("   python app.py")
    else:
        print("\n❌ No trained models found. You need to train first.")

if __name__ == "__main__":
    check_models() 
#!/usr/bin/env python3
"""
Quick setup script to use existing trained models
"""

import os
import shutil

def setup_existing_models():
    """Set up the app with existing trained models"""
    print("🐬 Setting up app with existing trained models...")
    
    # Check for existing models
    yolov8n_model = "../yolov8n/runs/detect_train/weights/best.pt"
    yolov8s_model = "../yolov8s/runs/detect_train/weights/best.pt"
    
    models_found = []
    
    if os.path.exists(yolov8n_model):
        print(f"✅ Found YOLOv8n model: {yolov8n_model}")
        models_found.append("YOLOv8n")
    else:
        print(f"❌ YOLOv8n model not found: {yolov8n_model}")
    
    if os.path.exists(yolov8s_model):
        print(f"✅ Found YOLOv8s model: {yolov8s_model}")
        models_found.append("YOLOv8s")
    else:
        print(f"❌ YOLOv8s model not found: {yolov8s_model}")
    
    if not models_found:
        print("❌ No trained models found. Please train models first.")
        return False
    
    print(f"\n📁 Found {len(models_found)} trained model(s): {', '.join(models_found)}")
    print("🚀 App is ready to use with existing models!")
    print("\n📱 To run the app:")
    print("   python app.py")
    print("\n🌐 The app will be available at: http://127.0.0.1:7860")
    
    return True

if __name__ == "__main__":
    setup_existing_models() 
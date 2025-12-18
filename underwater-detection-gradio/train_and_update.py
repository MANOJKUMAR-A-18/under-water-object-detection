#!/usr/bin/env python3
"""
Script to train both YOLOv8n and YOLOv8s models and update the Gradio app with the best one
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path

def run_training(model_type):
    """Run the training script for a specific model type"""
    print(f"🚀 Starting {model_type} model training...")
    
    # Path to the training script
    if model_type == "YOLOv8n":
        training_script = "train_and_test_yolov8.py"
        # Change to yolov8n directory for training
        original_dir = os.getcwd()
        os.chdir("../yolov8n")
    elif model_type == "YOLOv8s":
        training_script = "train_test_yolov8s.py"
        # Change to yolov8s directory for training
        original_dir = os.getcwd()
        os.chdir("../yolov8s")
    else:
        print(f"❌ Unknown model type: {model_type}")
        return False
    
    if not os.path.exists(training_script):
        print(f"❌ Training script not found: {training_script}")
        return False
    
    # Run the training script
    try:
        import subprocess
        result = subprocess.run([sys.executable, training_script], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {model_type} training completed successfully!")
        # Return to original directory
        os.chdir(original_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_type} training failed: {e}")
        print(f"Error output: {e.stderr}")
        # Return to original directory
        os.chdir(original_dir)
        return False

def get_model_performance(model_type):
    """Get performance metrics for a specific model"""
    if model_type == "YOLOv8n":
        results_file = "../yolov8n/runs/detect_val/results.csv"
    elif model_type == "YOLOv8s":
        results_file = "../yolov8s/runs/detect_val/results.csv"
    else:
        return None
    
    if not os.path.exists(results_file):
        return None
    
    try:
        df = pd.read_csv(results_file)
        if not df.empty:
            last_row = df.iloc[-1]
            map50 = last_row.get('metrics/mAP50(B)', last_row.get('metrics/mAP50', None))
            return map50
    except Exception as e:
        print(f"❌ Error reading performance for {model_type}: {e}")
    
    return None

def find_best_model():
    """Find the best performing model between YOLOv8n and YOLOv8s"""
    print("🔍 Comparing model performance...")
    
    performances = {}
    
    for model_type in ["YOLOv8n", "YOLOv8s"]:
        perf = get_model_performance(model_type)
        if perf is not None:
            performances[model_type] = perf
            print(f"   {model_type}: mAP50 = {perf:.3f}")
        else:
            print(f"   {model_type}: Performance data not available")
    
    if not performances:
        print("❌ No performance data available for any model")
        return None
    
    # Find the best model
    best_model = max(performances.items(), key=lambda x: x[1])
    print(f"🏆 Best model: {best_model[0]} (mAP50: {best_model[1]:.3f})")
    
    return best_model[0]

def copy_best_model_to_app(best_model_type):
    """Copy the best model to the app directory"""
    print(f"📁 Copying {best_model_type} model to app directory...")
    
    # Source model path
    source_model = f"../{best_model_type.lower()}/runs/detect_train/weights/best.pt"
    
    # Destination in app directory
    dest_model = f"best_{best_model_type.lower()}_model.pt"
    
    if not os.path.exists(source_model):
        print(f"❌ Trained model not found: {source_model}")
        return False
    
    try:
        shutil.copy2(source_model, dest_model)
        print(f"✅ Best model copied to: {dest_model}")
        return dest_model
    except Exception as e:
        print(f"❌ Failed to copy model: {e}")
        return False

def update_app_model_path(best_model_file):
    """Update the app.py to use the best model file"""
    print("🔧 Updating app.py to use best model...")
    
    app_file = "app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    try:
        # Read the current app.py
        with open(app_file, 'r') as f:
            content = f.read()
        
        # Update the model paths to use local files
        new_content = content.replace(
            '"YOLOv8n": "../yolov8n/runs/detect_train/weights/best.pt"',
            f'"YOLOv8n": "{best_model_file}"'
        )
        
        # Write back the updated content
        with open(app_file, 'w') as f:
            f.write(new_content)
        
        print("✅ App.py updated to use best model")
        return True
    except Exception as e:
        print(f"❌ Failed to update app.py: {e}")
        return False

def main():
    print("🐬 Underwater Detection Multi-Model Training & App Update")
    print("=" * 60)
    
    # Step 1: Train both models
    print("\n📚 Training both YOLOv8n and YOLOv8s models...")
    
    yolov8n_success = run_training("YOLOv8n")
    yolov8s_success = run_training("YOLOv8s")
    
    if not yolov8n_success and not yolov8s_success:
        print("❌ Both training runs failed. Exiting.")
        return
    
    # Step 2: Find the best model
    best_model_type = find_best_model()
    if not best_model_type:
        print("❌ Could not determine best model. Exiting.")
        return
    
    # Step 3: Copy best model to app directory
    best_model_file = copy_best_model_to_app(best_model_type)
    if not best_model_file:
        print("❌ Failed to copy best model. Exiting.")
        return
    
    # Step 4: Update app.py
    if not update_app_model_path(best_model_file):
        print("❌ Failed to update app. Exiting.")
        return
    
    print(f"\n🎉 Success! Your app is now ready with the best model ({best_model_type}).")
    print("📱 To run the app:")
    print("   python app.py")
    print(f"\n🏆 Using: {best_model_type} model")
    print("🌐 The app will be available at: http://127.0.0.1:7860")

if __name__ == "__main__":
    main() 
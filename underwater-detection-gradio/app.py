import gradio as gr
from PIL import Image
from ultralytics import YOLO
import os
import json

# Model paths for both YOLOv8n and YOLOv8s
MODEL_PATHS = {
    "YOLOv8n": "../yolov8n/runs/detect_train/weights/best.pt",
    "YOLOv8s": "../yolov8s/runs/detect_train/weights/best.pt"
}

# Load models
models = {}
available_models = []

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = YOLO(model_path)
            available_models.append(model_name)
            print(f"✅ Loaded {model_name} model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_name} model: {e}")
    else:
        print(f"⚠️ {model_name} model not found: {model_path}")

# Fallback to default model if no custom models found
if not available_models:
    models["Default YOLOv8n"] = YOLO("yolov8n.pt")
    available_models = ["Default YOLOv8n"]
    print("⚠️ No custom models found, using default YOLOv8n model")

# Underwater object classes from your training
UNDERWATER_CLASSES = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

def get_model_performance():
    """Get performance metrics for available models"""
    performance_info = {}
    
    for model_name in available_models:
        if model_name.startswith("Default"):
            performance_info[model_name] = "Default COCO model (not trained on underwater data)"
        else:
            # Try to read performance from validation results
            if "YOLOv8n" in model_name:
                results_file = "../yolov8n/runs/detect_val/results.csv"
            elif "YOLOv8s" in model_name:
                results_file = "../yolov8s/runs/detect_val/results.csv"
            else:
                results_file = None
            
            if results_file and os.path.exists(results_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        map50 = last_row.get('metrics/mAP50(B)', last_row.get('metrics/mAP50', 'N/A'))
                        performance_info[model_name] = f"mAP50: {map50:.3f}"
                    else:
                        performance_info[model_name] = "Training completed"
                except:
                    performance_info[model_name] = "Training completed"
            else:
                performance_info[model_name] = "Training completed"
    
    return performance_info

# Get model performance info
model_performance = get_model_performance()

# Create model selection with performance info and mapping
model_choices = []
display_to_key = {}
for model_name in available_models:
    perf_info = model_performance.get(model_name, "")
    display_name = f"{model_name} ({perf_info})" if perf_info else model_name
    model_choices.append(display_name)
    display_to_key[display_name] = model_name  # Map display name to model key

def detect_objects(image, model_choice, confidence_threshold=0.25):
    if image is None:
        return None, "Please upload an image"
    model_key = display_to_key.get(model_choice, None)
    if model_key not in models:
        return None, f"Model {model_choice} not available"
    model = models[model_key]
    results = model(image, conf=confidence_threshold)
    annotated = results[0].plot()
    detections = results[0]
    detection_info = []
    if detections.boxes is not None:
        for box in detections.boxes:
            if box.conf is not None and box.cls is not None:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                if model_key.startswith("Default"):
                    class_name = f"Object_{class_id}"
                else:
                    class_name = UNDERWATER_CLASSES[class_id] if class_id < len(UNDERWATER_CLASSES) else f"class_{class_id}"
                detection_info.append(f"{class_name}: {confidence:.2f}")
    detection_text = "\n".join(detection_info) if detection_info else "No objects detected"
    return Image.fromarray(annotated), detection_text

# Set YOLOv8n as the default value for the dropdown if available
default_model_choice = next((c for c in model_choices if c.lower().startswith('yolov8n')), model_choices[0] if model_choices else None)

# Supported languages
LANGUAGES = ["English", "French", "Spanish", "German", "Chinese"]

def build_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css="""
        footer {display: none !important;}
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        
        /* Dark mode (default) */
        :root {
            --bg-color: #1a1a1a;
            --text-color: #ffffff;
            --card-bg: #2d2d2d;
            --border-color: #404040;
        }
        
        /* Light mode */
        .light-mode {
            --bg-color: #ffffff;
            --text-color: #000000;
            --card-bg: #f5f5f5;
            --border-color: #e0e0e0;
        }
        
        body {
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
        }
        
        .gradio-container > div {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }
    """) as demo:
        gr.Markdown("""
        # Underwater Object Detection - Multi-Model Comparison
        Compare detection results using different YOLOv8 models. Available models: YOLOv8n, YOLOv8s. Supports: fish, jellyfish, penguin, puffin, shark, starfish, stingray.
        """)
        with gr.Accordion("Settings", open=False):
            light_mode = gr.Checkbox(label="Light Mode", value=False)
            language = gr.Dropdown(choices=LANGUAGES, value=LANGUAGES[0], label="Language")
            lang_label = gr.Markdown(f"**Selected Language:** {LANGUAGES[0]}")
            def update_language(selected):
                return f"**Selected Language:** {selected}"
            language.change(update_language, inputs=language, outputs=lang_label)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Underwater Image")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model_choice,
                    label="Select Model",
                    info="Choose the best performing model for your needs"
                )
                conf_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.25, 
                    step=0.05,
                    label="Confidence Threshold",
                    info="Adjust detection sensitivity"
                )
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Detection Results")
                output_text = gr.Textbox(label="Detected Objects", lines=5)
        submit_btn.click(
            detect_objects,
            inputs=[image_input, model_dropdown, conf_slider],
            outputs=[output_image, output_text]
        )
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch() 
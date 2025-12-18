#!/usr/bin/env python3
"""
Professional Underwater Detection System Launcher
Advanced launcher with system checks and professional features
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
import subprocess
import platform

def setup_logging():
    """Setup professional logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system_launcher.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('UnderwaterDetectionLauncher')

def check_system_requirements():
    """Check system requirements and dependencies"""
    logger = logging.getLogger('UnderwaterDetectionLauncher')
    
    logger.info("🔍 Checking system requirements...")
    
    # Python version check
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error("❌ Python 3.8+ required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version: %s", sys.version.split()[0])
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        logger.info("💾 Available RAM: %.1f GB", memory_gb)
        
        if memory_gb < 4:
            logger.warning("⚠️ Low memory detected. Recommended: 8GB+ RAM")
    except ImportError:
        logger.info("💾 Memory check skipped (psutil not available)")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("🎮 GPU detected: %s (Count: %d)", gpu_name, gpu_count)
        else:
            logger.info("🖥️ No GPU detected, using CPU")
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger = logging.getLogger('UnderwaterDetectionLauncher')
    
    required_packages = [
        'gradio', 'ultralytics', 'torch', 'opencv-python', 
        'Pillow', 'numpy', 'pandas', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info("✅ %s: installed", package)
        except ImportError:
            missing_packages.append(package)
            logger.error("❌ %s: missing", package)
    
    if missing_packages:
        logger.error("Missing packages: %s", ', '.join(missing_packages))
        logger.info("Run: pip install %s", ' '.join(missing_packages))
        return False
    
    return True

def check_models():
    """Check if detection models are available"""
    logger = logging.getLogger('UnderwaterDetectionLauncher')
    
    model_paths = {
        "YOLOv8n": "../yolov8n/runs/detect_train/weights/best.pt",
        "YOLOv8s": "../yolov8s/runs/detect_train/weights/best.pt"
    }
    
    fallback_models = ["yolov8n.pt", "yolov8s.pt"]
    
    available_models = []
    
    # Check custom trained models
    for name, path in model_paths.items():
        if os.path.exists(path):
            logger.info("✅ Found custom model: %s", name)
            available_models.append(name)
        else:
            logger.warning("⚠️ Custom model not found: %s", name)
    
    # Check fallback models
    for model in fallback_models:
        if os.path.exists(model):
            logger.info("✅ Found fallback model: %s", model)
        else:
            logger.info("📥 Fallback model will be downloaded: %s", model)
    
    if available_models:
        logger.info("🎯 Total available models: %d", len(available_models))
    else:
        logger.info("🎯 Will use default YOLO models")
    
    return True

def load_config():
    """Load professional configuration"""
    config_file = "config_professional.json"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Return default config
        return {
            "app_config": {
                "title": "Professional Underwater Detection System",
                "version": "2.0.0"
            }
        }

def print_banner(config):
    """Print professional banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🐬 {config['app_config']['title']:<63} ║
║                                                                              ║
║    Version: {config['app_config']['version']:<60} ║
║    Platform: {platform.system()} {platform.release():<54} ║
║    Python: {sys.version.split()[0]:<62} ║
║                                                                              ║
║    🎯 Advanced AI-powered underwater object detection                        ║
║    🚀 Real-time processing with professional analytics                       ║
║    📊 Multi-model comparison and performance tracking                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def launch_app(args):
    """Launch the professional application"""
    logger = logging.getLogger('UnderwaterDetectionLauncher')
    
    logger.info("🚀 Starting Professional Underwater Detection System...")
    
    # Determine which app to launch
    app_file = "app_professional.py" if args.professional else "app.py"
    
    if not os.path.exists(app_file):
        logger.error("❌ Application file not found: %s", app_file)
        return False
    
    # Set environment variables for optimization
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Launch command
    cmd = [sys.executable, app_file]
    
    if args.share:
        cmd.extend(['--share'])
    if args.port:
        cmd.extend(['--port', str(args.port)])
    
    try:
        logger.info("🌐 Application will be available at: http://localhost:%d", args.port or 7860)
        logger.info("📱 Starting application...")
        
        if args.professional:
            # Import and run professional app
            from app_professional import build_professional_app
            app = build_professional_app()
            app.launch(
                server_port=args.port or 7860,
                share=args.share,
                show_error=True,
                quiet=False
            )
        else:
            # Run standard app
            subprocess.run(cmd, check=True)
            
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error("❌ Failed to start application: %s", str(e))
        return False
    
    return True

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Professional Underwater Detection System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_professional.py                    # Launch basic version
  python launch_professional.py --professional    # Launch professional version
  python launch_professional.py --share --port 8080  # Share online on port 8080
  python launch_professional.py --gpu             # Enable GPU optimization
        """
    )
    
    parser.add_argument(
        '--professional', 
        action='store_true',
        help='Launch professional version with advanced features'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=7860,
        help='Port to run the application on (default: 7860)'
    )
    
    parser.add_argument(
        '--share', 
        action='store_true',
        help='Create public shareable link'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Enable GPU acceleration (if available)'
    )
    
    parser.add_argument(
        '--skip-checks', 
        action='store_true',
        help='Skip system requirement checks'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config()
    
    # Print banner
    print_banner(config)
    
    if not args.skip_checks:
        # Run system checks
        logger.info("🔧 Running system diagnostics...")
        
        if not check_system_requirements():
            logger.error("❌ System requirements check failed")
            return 1
        
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            logger.info("💡 Install missing dependencies with:")
            logger.info("    pip install -r requirements_professional.txt")
            return 1
        
        check_models()
        
        logger.info("✅ All system checks passed!")
    
    # Launch application
    success = launch_app(args)
    
    if success:
        logger.info("👋 Application finished successfully")
        return 0
    else:
        logger.error("❌ Application failed to start")
        return 1

if __name__ == "__main__":
    sys.exit(main())

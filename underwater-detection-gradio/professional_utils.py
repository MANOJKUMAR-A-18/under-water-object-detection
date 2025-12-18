"""
Professional utilities for underwater detection system
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any
import os

class DetectionAnalytics:
    """Professional analytics for detection results"""
    
    def __init__(self):
        self.session_data = []
    
    def add_detection(self, detection_data: Dict[str, Any]):
        """Add detection result to analytics"""
        self.session_data.append({
            **detection_data,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        if not self.session_data:
            return {}
        
        total_detections = len(self.session_data)
        total_objects = sum(len(d.get('objects', [])) for d in self.session_data)
        avg_inference_time = sum(d.get('inference_time', 0) for d in self.session_data) / total_detections
        
        # Class distribution
        class_counts = {}
        for detection in self.session_data:
            for obj in detection.get('objects', []):
                class_name = obj.get('class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'session_start': self.session_data[0]['timestamp'],
            'session_end': self.session_data[-1]['timestamp'],
            'total_images_processed': total_detections,
            'total_objects_detected': total_objects,
            'average_inference_time': avg_inference_time,
            'class_distribution': class_counts,
            'models_used': list(set(d.get('model', 'unknown') for d in self.session_data))
        }
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export detection data to CSV"""
        if filename is None:
            filename = f"detection_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.session_data:
            return filename
        
        # Flatten data for CSV
        csv_data = []
        for detection in self.session_data:
            base_row = {
                'timestamp': detection['timestamp'],
                'model': detection.get('model', ''),
                'inference_time': detection.get('inference_time', 0),
                'total_objects': len(detection.get('objects', []))
            }
            
            if detection.get('objects'):
                for obj in detection['objects']:
                    row = base_row.copy()
                    row.update({
                        'object_class': obj.get('class', ''),
                        'confidence': obj.get('confidence', 0),
                        'bbox_x1': obj.get('bbox', [0, 0, 0, 0])[0],
                        'bbox_y1': obj.get('bbox', [0, 0, 0, 0])[1],
                        'bbox_x2': obj.get('bbox', [0, 0, 0, 0])[2],
                        'bbox_y2': obj.get('bbox', [0, 0, 0, 0])[3]
                    })
                    csv_data.append(row)
            else:
                base_row.update({
                    'object_class': '',
                    'confidence': 0,
                    'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0
                })
                csv_data.append(base_row)
        
        # Write CSV
        if csv_data:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        return filename

class ModelPerformanceTracker:
    """Track and compare model performance"""
    
    def __init__(self):
        self.performance_data = {}
    
    def add_inference(self, model_name: str, inference_time: float, detection_count: int):
        """Add inference data point"""
        if model_name not in self.performance_data:
            self.performance_data[model_name] = {
                'inference_times': [],
                'detection_counts': [],
                'total_inferences': 0
            }
        
        self.performance_data[model_name]['inference_times'].append(inference_time)
        self.performance_data[model_name]['detection_counts'].append(detection_count)
        self.performance_data[model_name]['total_inferences'] += 1
    
    def get_model_stats(self, model_name: str) -> Dict[str, float]:
        """Get statistics for a specific model"""
        if model_name not in self.performance_data:
            return {}
        
        data = self.performance_data[model_name]
        
        return {
            'avg_inference_time': sum(data['inference_times']) / len(data['inference_times']),
            'min_inference_time': min(data['inference_times']),
            'max_inference_time': max(data['inference_times']),
            'avg_detections': sum(data['detection_counts']) / len(data['detection_counts']),
            'total_inferences': data['total_inferences'],
            'fps': 1.0 / (sum(data['inference_times']) / len(data['inference_times']))
        }
    
    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """Compare all models"""
        comparison = {}
        for model_name in self.performance_data:
            comparison[model_name] = self.get_model_stats(model_name)
        return comparison

class ProfessionalReporting:
    """Generate professional reports"""
    
    @staticmethod
    def generate_html_report(analytics: DetectionAnalytics, 
                           performance_tracker: ModelPerformanceTracker) -> str:
        """Generate a comprehensive HTML report"""
        
        session_summary = analytics.get_session_summary()
        model_comparison = performance_tracker.compare_models()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Underwater Detection Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', sans-serif; 
                    margin: 40px; 
                    background: #f8f9fa; 
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 2rem; 
                    border-radius: 10px; 
                    text-align: center;
                    margin-bottom: 2rem;
                }}
                .section {{ 
                    background: white; 
                    padding: 1.5rem; 
                    margin: 1rem 0; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metric {{ 
                    display: inline-block; 
                    margin: 0.5rem; 
                    padding: 1rem; 
                    background: #e9ecef; 
                    border-radius: 5px; 
                    min-width: 150px;
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #007bff; 
                }}
                .metric-label {{ 
                    color: #6c757d; 
                    font-size: 0.9em; 
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 1rem 0; 
                }}
                th, td {{ 
                    padding: 0.75rem; 
                    text-align: left; 
                    border-bottom: 1px solid #dee2e6; 
                }}
                th {{ 
                    background-color: #e9ecef; 
                    font-weight: 600; 
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🐬 Underwater Detection System Report</h1>
                <p>Professional Analysis & Performance Metrics</p>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>📊 Session Summary</h2>
                <div class="metric">
                    <div class="metric-value">{total_images}</div>
                    <div class="metric-label">Images Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_objects}</div>
                    <div class="metric-label">Objects Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_time:.2f}s</div>
                    <div class="metric-label">Avg Inference Time</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Object Distribution</h2>
                <table>
                    <thead>
                        <tr><th>Object Class</th><th>Count</th><th>Percentage</th></tr>
                    </thead>
                    <tbody>
                        {class_distribution_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🤖 Model Performance</h2>
                <table>
                    <thead>
                        <tr><th>Model</th><th>Avg Time (s)</th><th>FPS</th><th>Avg Detections</th></tr>
                    </thead>
                    <tbody>
                        {model_performance_rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Format data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_images = session_summary.get('total_images_processed', 0)
        total_objects = session_summary.get('total_objects_detected', 0)
        avg_time = session_summary.get('average_inference_time', 0)
        
        # Class distribution rows
        class_dist = session_summary.get('class_distribution', {})
        total_class_count = sum(class_dist.values()) if class_dist else 1
        class_rows = ""
        for class_name, count in class_dist.items():
            percentage = (count / total_class_count) * 100
            class_rows += f"<tr><td>{class_name}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        # Model performance rows  
        perf_rows = ""
        for model_name, stats in model_comparison.items():
            perf_rows += f"""<tr>
                <td>{model_name}</td>
                <td>{stats.get('avg_inference_time', 0):.3f}</td>
                <td>{stats.get('fps', 0):.1f}</td>
                <td>{stats.get('avg_detections', 0):.1f}</td>
            </tr>"""
        
        return html_template.format(
            timestamp=timestamp,
            total_images=total_images,
            total_objects=total_objects,
            avg_time=avg_time,
            class_distribution_rows=class_rows,
            model_performance_rows=perf_rows
        )

# Global instances
analytics = DetectionAnalytics()
performance_tracker = ModelPerformanceTracker()
reporting = ProfessionalReporting()

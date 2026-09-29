# YOLO Object Detection and Traffic Analysis

## Overview

This project provides an end-to-end object detection, tracking, and performance analysis pipeline utilizing Ultralytics YOLO models (specifically YOLO11 architectures: nano, small, and medium). The pipeline evaluates detection performance across various road conditions, environmental lighting scenarios, vehicle density levels, and video tracking workflows.

## Features

- Multi-Class Object Detection: Detects vehicles (cars, buses, trucks, motorcycles, trains), pedestrians, and traffic infrastructure (traffic lights) with high precision.
- Environmental Scenario Benchmark: Compares detection confidence and throughput across Daylight, Heavy Traffic, Night, Fog, Rain, and Pedestrian Scene conditions.
- Model Performance Comparison: Benchmarks YOLO11n (nano), YOLO11s (small), and YOLO11m (medium) for object count, inference latency, and frames per second (FPS).
- Real-Time Video Object Tracking: Integrates ByteTrack with YOLO11 for tracking multi-vehicle traffic in video feeds.
- Automated Metric Export: Automatically aggregates and exports evaluation data into structured CSV reports.

## Project Structure

```text
Yolo/
│
├── data/                               # Dataset files
│   ├── images/                         # Road and traffic test images (road1.jpg - road14.jpg)
│   └── videos/                         # Input traffic video feeds
│
├── notebooks/                          # Jupyter Notebook workflows
│   └── YOLO_Object_Detection_Analysis.ipynb
│
├── outputs/                            # Pipeline output demonstrations
│   ├── images/                         # Saved detection images
│   └── videos/                         # Processed tracking output videos (H264)
│
├── results/                            # Structured experimental outputs
│   ├── figures/                        # Generated charts and evaluation plots
│   │   ├── accuracy_metrics_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── loss_curves.png
│   │   └── train_test_training_graph.png
│   └── metrics/                        # Quantitative CSV benchmark results
│       ├── final_model_comparison.csv
│       ├── yolo_detection_results.csv
│       ├── yolo_environment_analysis.csv
│       ├── yolo_model_comparison.csv
│       ├── yolo_object_analysis.csv
│       └── yolo_realtime_performance.csv
│
├── src/                                # Source modules
│   ├── __init__.py
│   └── evaluation_metrics.py           # Evaluation, confusion matrix, and plotting suite
│
├── main.py                             # Main CLI entrypoint
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation
└── PROJECT_REPORT.md                   # Technical project report
```

## Installation

1. Ensure Python 3.8 or newer is installed on your system.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

- ultralytics: YOLO object detection and tracking framework
- torch and torchvision: Deep learning tensor processing and GPU acceleration
- opencv-python: Image and video processing
- pandas: Data manipulation and tabular results analysis
- matplotlib and seaborn: Visualization, heatmap, and chart generation
- scikit-learn: Metric computation and confusion matrix generation
- Pillow: Image file loading and manipulation
- ipython and jupyter: Interactive notebook execution

## Usage

### 1. Unified Command Line Interface

Execute the main CLI entrypoint for detection, tracking, or evaluation:

```bash
# Run evaluation metrics and generate all charts
python main.py --mode evaluate

# Run detection on a test image
python main.py --mode detect --image data/images/road1.jpg

# Run ByteTrack multi-object tracking on video
python main.py --mode track --video "data/videos/Traffic sound, Indian Traffic Sounds, Traffic Noise, Vehicle Noise, Road traffic#Shorts#Viral #cars - 720.mp4"
```

### 2. Running the Jupyter Notebook

Launch the notebook interface:

```bash
jupyter notebook notebooks/YOLO_Object_Detection_Analysis.ipynb
```

### 3. Generating Evaluation Metrics Directly

Run the dedicated evaluation module to generate figures in `results/figures/`:

```bash
python src/evaluation_metrics.py
```

### 4. Standalone Python Inference Example

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11n.pt")

# Perform inference on an image
results = model("data/images/road1.jpg")

# Display results
for r in results:
    r.show()

# Run tracking on a video
model.track(
    source="data/videos/Traffic sound, Indian Traffic Sounds, Traffic Noise, Vehicle Noise, Road traffic#Shorts#Viral #cars - 720.mp4",
    tracker="bytetrack.yaml",
    save=True,
    project="outputs/videos"
)
```

## Performance Benchmarks

### Model Comparison Summary

| Model | Average Objects Detected | Average Confidence | Average Inference Time (ms) | Average FPS |
| :--- | :--- | :--- | :--- | :--- |
| YOLO11n | 13.6 | 0.447 | 15.23 ms | 66.2 FPS |
| YOLO11s | 9.0 | 0.455 | 19.17 ms | 53.0 FPS |
| YOLO11m | 10.2 | 0.434 | 21.99 ms | 46.9 FPS |

### Environment Analysis Summary

| Environment | Images | Total Objects | Average Confidence (%) |
| :--- | :--- | :--- | :--- |
| Daylight | 5 | 88 | 49.20% |
| Heavy Traffic | 3 | 75 | 48.06% |
| Night | 2 | 23 | 27.61% |
| Fog | 1 | 16 | 54.94% |
| Pedestrian Scene | 2 | 41 | 47.10% |
| Rain | 1 | 2 | 74.74% |

## License

This project is open-source and available under the MIT License.

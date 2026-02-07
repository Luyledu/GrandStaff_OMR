End-to-End Grand Staff Optical Music Recognition System
🎯 Project Overview
This system implements a complete sheet music recognition pipeline, integrating:

1.YOLO11m segmentation – Detects grand staff regions

2.Post-processing – Corrects tilt and expands regions

3.Size standardization – Resizes to target height

4.Sheet Music Transformer recognition – Recognizes musical symbols

5.Result integration – Combines recognition results from all regions

🚀 Quick Start
1. Environment Setup
bash
# Install dependencies  
pip install -r requirements.txt  

# Install PyTorch (if using GPU)  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  
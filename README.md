# GrandStaff_OMR
GrandStaff OMR System is an end-to-end Optical Music Recognition (OMR) system specifically designed for grand staff notation. The system automatically detects, processes, and recognizes musical symbols from scanned sheet music images, outputting structured music notation data.

✨ Key Features
1.Complete Pipeline: Integrated system from image input to music notation output
2.Robust Detection: YOLO11m-based grand staff region detection
3.Intelligent Processing: Automatic tilt correction and region expansion
4.Advanced Recognition: Sheet Music Transformer for accurate symbol recognition
5.Standardized Output: Consistent formatting across different input resolutions
6.Multi-Region Support: Handles multiple grand staff systems in a single image

🔧 How It Works
The system follows a sophisticated five-step process:
1.Staff Detection: YOLO11m model identifies grand staff regions in input images
2.Image Processing: Corrects tilt, extends regions, and prepares for recognition
3.Size Standardization: Normalizes images to optimal dimensions for recognition
4.Symbol Recognition: Sheet Music Transformer analyzes and identifies musical symbols
5.Result Integration: Combines recognized symbols into coherent music notation

🚀 Quick Installation
bash
# Clone repository
git clone https://github.com/yourusername/GrandStaff_OMR.git
cd grandstaff-omr

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (GPU version recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Acknowledgments
Original YOLO implementation by Ultralytics
Sheet Music Transformer research teams
Open-source music notation libraries
Contributors and testers

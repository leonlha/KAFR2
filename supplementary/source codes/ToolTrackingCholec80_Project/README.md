# Surgical Tool Tracking

# Installation
- Python 3.9 (from anaconda navigator)
- Numpy
- Microsof C++ Build Tools
- Install lap
- Install requirement.txt
- Install torch gpu: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Run
- C:\ProgramData\anaconda3\Scripts\activate tool
- python 1_tracking.py --g=0 --r=D:\Research\ToolTrackingCholet80
- change folder names using doing.ipynb (option)
- python 2_centroid-afr.py

# Train/Test split
- https://github.com/YuemingJin/MTRCNet-CL/blob/master/get_paths_labels.py

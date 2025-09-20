## Description

Source code for identifying the phases for robotic GJ videos.


## Requirements 


```shell
pip install -r requirements.txt
```


## Usage

1. Extract, crop and resize images of all videos 
2. Plot and analyze the annotations
3. Create train-val-test files (txt format) containing the list of images as well as the length and frame rate of the video
4. Train X3D model
5. Make predictions
6. Run temporal smoothing
7. Extract images for Optical Flow, repeat steps 4-6, ensemble two models
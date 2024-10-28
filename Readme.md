# Image Processing System

## Overview

This project contains C++ code designed to handle 16-bit image data processing from a `txr_CBpDetectorImgBuf` object, which represents image slices captured from a detector. The code processes these images in specific ways, such as discarding certain images if they contain mostly background pixels and splitting images based on time differences between slices. The key functionalities include reading images, determining if an image should be discarded, and processing image slices.

## Features

- **Read 16-bit Image Data**: Extracts 16-bit image data from the `txr_CBpDetectorImgBuf` buffer into an OpenCV `cv::Mat` matrix for further processing.
- **Discard Unnecessary Images**: Evaluates whether an image should be discarded based on its pixel composition. If over 95% of the pixels are zero, the image is considered mostly blank and can be discarded.
- **Slice Processing**: Retains only the first 12 columns of image slices that meet the blank image criteria.
- **Time and Pixel-Based Image Splitting**: The code compares the time difference between image slices and the pixel data to decide if an image needs to be split for further processing.

## Usage
To run the code, ensure that you have OpenCV installed. Compile the code using a C++ compiler, linking OpenCV libraries as shown below:
```cpp
g++ -o image_processing main.cpp `pkg-config --cflags --libs opencv4`
```
## License
This project is licensed under the MIT License.


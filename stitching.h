#pragma once
#include <opencv2/opencv.hpp>
#include <queue>

// ”Ÿ”º–¾
std::pair<cv::Mat, cv::Mat> bufferAndStitchImages(
    const cv::Mat& high_before,
    const cv::Mat& high_restart,
    const cv::Mat& low_before,
    const cv::Mat& low_restart);


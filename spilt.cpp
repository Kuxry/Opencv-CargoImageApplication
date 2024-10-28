#include "txr_CBpDetectorImgBuf.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>
//test
// 传送带速度（米/秒）
const double conveyorSpeed = 1.0;  // 假设传送带速度为 1 米/秒
const int sliceIntervalMicroseconds = 500;  // 每个切片的时间间隔 500 微秒


// 从 txr_CBpDetectorImgBuf 中读取16位图像数据
cv::Mat read16BitImageFromTxrBuf(txr_CBpDetectorImgBuf* imgBuf, int width, int height) {
    if (imgBuf == nullptr) {
        std::cerr << "图像缓冲区为空" << std::endl;
        return cv::Mat();
    }

    // 将 denseImageData 中的数据转为 cv::Mat
    cv::Mat img(height, width, CV_16UC1, imgBuf->denseImageData.data());
    if (img.empty()) {
        std::cerr << "图像创建失败" << std::endl;
    }

    return img;
}

// 判断图片是否需要舍弃（像素值80%以上为0），使用更高效的OpenCV函数
bool shouldDiscardImage(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "空图像，无法进行判断" << std::endl;
        return false;
    }

    int totalPixels = image.rows * image.cols;
    int nonZeroPixels = cv::countNonZero(image);
    int zeroPixels = totalPixels - nonZeroPixels;

    double zeroRatio = static_cast<double>(zeroPixels) / totalPixels;

    // 如果80%以上的像素是0，则返回true，表示需要舍弃
    return zeroRatio >= 0.95;
}

// 处理留白切片，只保留前12列，舍弃后12列
void processBlankSlice(cv::Mat& image, int columnsToKeep = 12) {
    if (image.empty()) return;

    // 确保 columnsToKeep 不超过图像宽度
    columnsToKeep = std::min(columnsToKeep, image.cols);

    // 遍历每一行，保留前12列，舍弃后12列
    for (int row = 0; row < image.rows; ++row) {
        for (int col = columnsToKeep; col < image.cols; ++col) {
            image.at<ushort>(row, col) = 0; // 或者你可以选择其他方式处理舍弃的部分
        }
    }
}

// 根据传送带时间差和图像内容进行判断并分割图像
void splitImageBasedOnTimeAndPixels(txr_CBpDetectorImgBuf* imgBuf, int width, int height, std::chrono::microseconds timeDifference) {
    int totalSlices = 60;  // 假设一张图片由60个切片组成
    int totalMicrosecondsPerImage = totalSlices * sliceIntervalMicroseconds;

    // 从 denseImageData 中读取图像数据
    cv::Mat densityImage = read16BitImageFromTxrBuf(imgBuf, width, height);

    // 判断是否需要进行图像切割：时间差 + 像素值判断
    if (timeDifference.count() > totalMicrosecondsPerImage && shouldDiscardImage(densityImage)) {
        std::cout << "时间差和像素值条件都满足，进行图像切割。" << std::endl;

        // 切分图像，将 denseImageData 分成两部分
        int middleIndex = imgBuf->denseImageData.size() / 2;

        // 创建新的图像对象（假设这里只处理一半）
        std::vector<uint16_t> firstHalf(imgBuf->denseImageData.begin(), imgBuf->denseImageData.begin() + middleIndex);
        std::vector<uint16_t> secondHalf(imgBuf->denseImageData.begin() + middleIndex, imgBuf->denseImageData.end());

        // 处理两部分图像
        std::cout << "处理第一部分图像..." << std::endl;

        imgBuf->denseImageData = firstHalf;

    }
    else {
        std::cout << "时间差或像素值条件不满足，不进行图像切割。" << std::endl;
    }
}

int main() {
    int width = 2304;  // 假设图像宽度为2304列
    int height = 24;   // 假设图像高度为24行

    // 创建 txr_CBpDetectorImgBuf 对象并填充数据
    auto imgBuf = std::make_unique<txr_CBpDetectorImgBuf>(16, width, height);

    // 初始化 denseImageData，填充一些示例数据
    imgBuf->denseImageData = std::vector<uint16_t>(width * height, 100);

    // 为了测试，将前80%的像素值设为0
    int numZeroPixels = static_cast<int>(0.80 * width * height);
    for (int i = 0; i < numZeroPixels; ++i) {
        imgBuf->denseImageData[i] = 0;
    }

    // 假设前后两次切片的时间差为500微秒
    std::chrono::microseconds timeDifference(500);

    // 判断并根据时间差和像素值进行图像切割
    splitImageBasedOnTimeAndPixels(imgBuf.get(), width, height, timeDifference);

    return 0;
}

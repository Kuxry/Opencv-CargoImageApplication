#include "txr_CBpDetectorImgBuf.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>

// 传送带速度（米/秒）
const double conveyorSpeed = 1.0;  // 假设传送带速度为 1 米/秒
const int sliceIntervalMicroseconds = 500;  // 每个切片的时间间隔 500 微秒
const int sliceWidth = 24;  // 切片的宽度为24

// 从PNG图像文件读取图像数据并切割成多个切片
std::vector<cv::Mat> readImageSlicesFromFile(const std::string& filePath) {
    // 使用OpenCV读取PNG文件
    cv::Mat img = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    std::vector<cv::Mat> imageSlices;

    if (img.empty()) {
        std::cerr << "无法读取图片: " << filePath << std::endl;
        return imageSlices;
    }

    // 将图像切割成宽度为sliceWidth的多个切片
    int numSlices = img.cols / sliceWidth;  // 计算有多少个切片
    for (int i = 0; i < numSlices; ++i) {
        // 使用OpenCV的裁剪功能
        cv::Rect sliceROI(i * sliceWidth, 0, sliceWidth, img.rows);
        imageSlices.push_back(img(sliceROI).clone());  // 克隆切片数据到vector中
    }

    return imageSlices;
}

// 从 txr_CBpDetectorImgBuf 中读取图像数据
cv::Mat readImageFromTxrBuf(txr_CBpDetectorImgBuf* imgBuf, int width, int height) {
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

// 判断图片是否需要舍弃（像素值80%以上为0）
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
    cv::Mat densityImage = readImageFromTxrBuf(imgBuf, width, height);

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
    int height = 24;   // 切片的高度

    // 从文件中读取并切片 PNG 图片
    std::string filePath = "image.png";
    std::vector<cv::Mat> imageSlices = readImageSlicesFromFile(filePath);

    if (imageSlices.empty()) {
        std::cerr << "读取切片失败，程序结束。" << std::endl;
        return -1;
    }

    // 创建 txr_CBpDetectorImgBuf 对象并填充数据
    auto imgBuf = std::make_unique<txr_CBpDetectorImgBuf>(16, sliceWidth, height);

    // 假设前后两次切片的时间差为500微秒
    std::chrono::microseconds timeDifference(500);

    // 遍历每个切片，依次进行处理
    for (const auto& slice : imageSlices) {
        // 将图像数据转换为 uint16_t 并存储在 denseImageData 中
        imgBuf->denseImageData.assign((uint16_t*)slice.data, (uint16_t*)slice.data + slice.total());

        // 判断并根据时间差和像素值进行图像切割
        splitImageBasedOnTimeAndPixels(imgBuf.get(), slice.cols, slice.rows, timeDifference);
    }

    return 0;
}

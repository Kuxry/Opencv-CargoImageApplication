#include "txr_CBpDetectorImgBuf.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>

const int sliceIntervalMicroseconds = 500;  // 每个切片的时间间隔 500 微秒
const int sliceWidth = 24;  // 切片的宽度为24  ccvv

// 从PNG图像文件读取图像数据并切割成多个切片
std::vector<cv::Mat> readImageSlicesFromFile(const std::string& filePath) {
    // 使用OpenCV读取PNG文件
    cv::Mat img = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    std::vector<cv::Mat> imageSlices;

    // 检查图像是否成功加载
    if (img.empty()) {
        std::cerr << "无法读取图片: " << filePath << std::endl;
        return imageSlices;
    }

    // 计算可以切割的完整切片数量
    int numSlices = img.cols / sliceWidth;
    int remainder = img.cols % sliceWidth;  // 计算剩余的像素宽度

    // 将图像切割成多个宽度为sliceWidth的切片
    for (int i = 0; i < numSlices; ++i) {
        // 定义当前切片的区域
        cv::Rect sliceROI(i * sliceWidth, 0, sliceWidth, img.rows);
        // 克隆切片数据到vector中
        imageSlices.push_back(img(sliceROI).clone());
    }

    // 如果有剩余像素，创建一个宽度为remainder的额外切片
    if (remainder > 0) {
        cv::Rect remainderROI(numSlices * sliceWidth, 0, remainder, img.rows);
        imageSlices.push_back(img(remainderROI).clone());
    }

    return imageSlices;
}

// 从 txr_CBpDetectorImgBuf 中读取图像数据并转换为OpenCV矩阵
cv::Mat readImageFromTxrBuf(txr_CBpDetectorImgBuf* imgBuf, int width, int height) {
    // 检查图像缓冲区是否为空
    if (imgBuf == nullptr) {
        std::cerr << "图像缓冲区为空" << std::endl;
        return cv::Mat();
    }

    // 将 denseImageData 中的数据转为 OpenCV 矩阵
    cv::Mat img(height, width, CV_16UC1, imgBuf->denseImageData.data());
    if (img.empty()) {
        std::cerr << "图像创建失败" << std::endl;
    }

    return img;
}

// 判断图片是否需要舍弃（如果95%以上的像素为0则舍弃）
bool shouldDiscardImage(const cv::Mat& image) {
    // 检查图像是否为空
    if (image.empty()) {
        std::cerr << "空图像，无法进行判断" << std::endl;
        return false;
    }

    // 计算总像素数、非零像素数及零像素数
    int totalPixels = image.rows * image.cols;
    int nonZeroPixels = cv::countNonZero(image);
    int zeroPixels = totalPixels - nonZeroPixels;

    // 计算零像素比例
    double zeroRatio = static_cast<double>(zeroPixels) / totalPixels;

    // 如果零像素比例大于等于95%，则返回true
    return zeroRatio >= 0.95;
}

// 根据图像内容和动态分割位置进行判断并分割图像
void splitImageBasedOnTimeAndPixels(txr_CBpDetectorImgBuf* imgBuf, int width, int height, int sliceIndex, int totalSlices) {
    // 从密集图像数据中读取图像
    cv::Mat densityImage = readImageFromTxrBuf(imgBuf, width, height);

    // 动态计算分割触发位置（默认为总切片数的80%处）
    int splitTriggerIndex = static_cast<int>(totalSlices * 0.5);

    // 满足丢弃条件且达到分割位置时触发分割
    if (shouldDiscardImage(densityImage) && sliceIndex == splitTriggerIndex) {
        std::cout << "满足条件，触发分割操作。" << std::endl;

        // 分割 denseImageData，将数据分为前半部分
        int middleIndex = imgBuf->denseImageData.size() / 2;
        std::vector<uint16_t> firstHalf(imgBuf->denseImageData.begin(), imgBuf->denseImageData.begin() + middleIndex);

        // 将 denseImageData 设置为第一部分（删去后半部分）
        imgBuf->denseImageData = firstHalf;
    }
    else {
        std::cout << "条件不满足，不触发分割。" << std::endl;
    }
}

int main() {
    std::string filePath = "image.png";  // 图像文件路径
    // 从文件中读取并切片 PNG 图片
    std::vector<cv::Mat> imageSlices = readImageSlicesFromFile(filePath);

    // 检查是否成功读取切片
    if (imageSlices.empty()) {
        std::cerr << "读取切片失败" << std::endl;
        return -1;
    }

    int totalSlices = imageSlices.size();  // 总切片数量
    int sliceIndex = 0;  // 当前切片索引

    // 遍历每个切片，依次进行处理
    for (const auto& slice : imageSlices) {
        // 动态获取图像的位深
        int bitDepth = slice.elemSize() * 8;

        // 创建 txr_CBpDetectorImgBuf 对象并填充数据
        auto imgBuf = std::make_unique<txr_CBpDetectorImgBuf>(bitDepth, slice.cols, slice.rows);

        // 将图像数据转换为 uint16_t 并存储在 denseImageData 中
        imgBuf->denseImageData.assign((uint16_t*)slice.data, (uint16_t*)slice.data + slice.total());

        // 调用 splitImageBasedOnTimeAndPixels 函数，根据图像内容和切片位置判断是否需要分割
        splitImageBasedOnTimeAndPixels(imgBuf.get(), slice.cols, slice.rows, sliceIndex, totalSlices);

        // 如果进行了图像切割且 denseImageData 仍有内容，保存该图像切片
        if (imgBuf->denseImageData.size() != slice.total()) {
            std::string fileName = "processed_slice_" + std::to_string(sliceIndex) + ".png";
            cv::Mat processedSlice(slice.rows, slice.cols, CV_16UC1, imgBuf->denseImageData.data());
            cv::imwrite(fileName, processedSlice);
            std::cout << "处理后的切片 " << sliceIndex << " 保存为 " << fileName << std::endl;
        }

        sliceIndex++;  // 更新切片索引
    }

    return 0;
}

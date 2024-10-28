#include "txr_CBpDetectorImgBuf.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>



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

// 判断图片是否需要舍弃（像素值95%以上为0）
bool shouldDiscardImage(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "空图像，无法进行判断" << std::endl;
        return false;
    }

    int totalPixels = image.rows * image.cols;
    int nonZeroPixels = cv::countNonZero(image);
    int zeroPixels = totalPixels - nonZeroPixels;

    double zeroRatio = static_cast<double>(zeroPixels) / totalPixels;

    // 如果95%以上的像素是0，则返回true，表示需要舍弃
    return zeroRatio >= 0.95;
}

// 根据图像内容进行判断并分割图像
void splitImageBasedOnTimeAndPixels(txr_CBpDetectorImgBuf* imgBuf, int width, int height, int sliceIndex) {
    // 从 denseImageData 中读取图像数据
    cv::Mat densityImage = readImageFromTxrBuf(imgBuf, width, height);

    // 判断是否需要进行图像切割：基于像素值判断或者到达第48张切片时进行分割
    if (shouldDiscardImage(densityImage) && sliceIndex == 48) {
        std::cout << "条件满足，进行图像切割。" << std::endl;

        // 切分图像，将 denseImageData 分成两部分
        int middleIndex = imgBuf->denseImageData.size() / 2;

        // 创建新的图像对象（处理第一部分）
        std::vector<uint16_t> firstHalf(imgBuf->denseImageData.begin(), imgBuf->denseImageData.begin() + middleIndex);

        // 将 denseImageData 设置为第一部分
        imgBuf->denseImageData = firstHalf;
    }
    else {
        std::cout << "条件不满足，不进行图像切割。" << std::endl;
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

    // 遍历每个切片，依次进行处理
    int sliceIndex = 0;
    for (const auto& slice : imageSlices) {
        // 动态获取图像的位深
        int bitDepth = slice.elemSize() * 8;

        // 创建 txr_CBpDetectorImgBuf 对象并填充数据
        auto imgBuf = std::make_unique<txr_CBpDetectorImgBuf>(bitDepth, slice.cols, slice.rows);

        // 将图像数据转换为 uint16_t 并存储在 denseImageData 中
        imgBuf->denseImageData.assign((uint16_t*)slice.data, (uint16_t*)slice.data + slice.total());

        // 调用 splitImageBasedOnTimeAndPixels 函数，判断是否需要切割
        splitImageBasedOnTimeAndPixels(imgBuf.get(), slice.cols, slice.rows, sliceIndex);

        // 如果进行了图像切割并且 denseImageData 还有内容，才保存图像
        if (imgBuf->denseImageData.size() != slice.total()) {
            std::string fileName = "processed_slice_" + std::to_string(sliceIndex) + ".png";
            cv::Mat processedSlice(slice.rows, slice.cols, CV_16UC1, imgBuf->denseImageData.data());
            cv::imwrite(fileName, processedSlice);
            std::cout << "处理后的切片 " << sliceIndex << " 保存为 " << fileName << std::endl;
        }

        sliceIndex++;
    }

    return 0;
}

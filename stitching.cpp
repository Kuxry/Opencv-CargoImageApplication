#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include "stitching.h"


// SIFT特征点检测和描述子提取
std::pair<std::vector<cv::KeyPoint>, cv::Mat> sift_feature_detector_and_descriptor(const cv::Mat& image) {
    // 创建SIFT对象
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // SIFT特征点和描述子
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // 检测特征点和计算描述子
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return { keypoints, descriptors };
}

// 特征点匹配
std::vector<cv::DMatch> feature_matching(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    // 创建BFMatcher对象，使用L2范数和交叉检查
    cv::BFMatcher bf(cv::NORM_L2, true);

    // 进行匹配
    std::vector<cv::DMatch> matches;
    bf.match(descriptors1, descriptors2, matches);

    // 按距离排序
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
        });

    return matches;
}

// RANSAC计算单应矩阵
std::pair<cv::Mat, cv::Mat> ransac_homography(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches) {
    if (matches.size() < 4) {
        return { cv::Mat(), cv::Mat() };
    }

    // 获取匹配点对的坐标
    std::vector<cv::Point2f> src_pts, dst_pts;
    for (const auto& match : matches) {
        src_pts.push_back(kp1[match.queryIdx].pt);
        dst_pts.push_back(kp2[match.trainIdx].pt);
    }

    // 使用RANSAC计算单应矩阵
    cv::Mat mask;
    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0, mask);

    return { H, mask };
}

// 图像拼接
cv::Mat stitch_images(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& H) {
    // 获取图像的尺寸
    int h1 = image1.rows, w1 = image1.cols;
    int h2 = image2.rows, w2 = image2.cols;

    // 计算拼接后图像的大小
    std::vector<cv::Point2f> corners_image1 = { cv::Point2f(0, 0), cv::Point2f(0, h1), cv::Point2f(w1, h1), cv::Point2f(w1, 0) };
    std::vector<cv::Point2f> corners_image2 = { cv::Point2f(0, 0), cv::Point2f(0, h2), cv::Point2f(w2, h2), cv::Point2f(w2, 0) };
    std::vector<cv::Point2f> transformed_corners_image1;

    cv::perspectiveTransform(corners_image1, transformed_corners_image1, H);
    std::vector<cv::Point2f> all_corners = transformed_corners_image1;
    all_corners.insert(all_corners.end(), corners_image2.begin(), corners_image2.end());

    cv::Point2f x_min = *std::min_element(all_corners.begin(), all_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
        });
    cv::Point2f y_min = *std::min_element(all_corners.begin(), all_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
        });
    cv::Point2f x_max = *std::max_element(all_corners.begin(), all_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x < b.x;
        });
    cv::Point2f y_max = *std::max_element(all_corners.begin(), all_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
        });

    cv::Point2f translation_dist(-x_min.x, -y_min.y);
    cv::Mat H_translation = (cv::Mat_<double>(3, 3) << 1, 0, translation_dist.x, 0, 1, translation_dist.y, 0, 0, 1);

    cv::Size result_size(x_max.x - x_min.x, y_max.y - y_min.y);

    cv::Mat stitched_image;
    cv::warpPerspective(image2, stitched_image, H_translation * H, result_size);

    // 将 image1 放到 result 中正确的位置
    cv::Mat roi(stitched_image, cv::Rect(translation_dist.x, translation_dist.y, w1, h1));
    image1.copyTo(roi);

    // 去除黑色部分
    cv::Mat stitched_image_8bit;
    if (stitched_image.depth() == CV_16U) {
        stitched_image.convertTo(stitched_image_8bit, CV_8U, 255.0 / 65535.0); // 图像16bit to 8bit
    }
    else {
        stitched_image_8bit = stitched_image;
    }

    cv::Mat thresh;
    cv::threshold(stitched_image_8bit, thresh, 1, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect bounding_rect = cv::boundingRect(contours[0]);
    stitched_image = stitched_image(bounding_rect);

    return stitched_image;
}

// 高低能图像拼接
std::pair<cv::Mat, cv::Mat> stitch_dual_energy_images(const cv::Mat& high_energy_before_stop, const cv::Mat& high_energy_after_restart, const cv::Mat& low_energy_before_stop, const cv::Mat& low_energy_after_restart) {
    // 判断输入图像状态
    if (high_energy_before_stop.empty() || high_energy_after_restart.empty() || low_energy_before_stop.empty() || low_energy_after_restart.empty()) {
        std::cerr << "One or more images are empty!" << std::endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }

    if (high_energy_before_stop.type() != CV_16UC1 || high_energy_after_restart.type() != CV_16UC1 || low_energy_before_stop.type() != CV_16UC1 || low_energy_after_restart.type() != CV_16UC1) {
        std::cerr << "Error: Image is not a 16-bit single channel (grayscale) image!" << std::endl;
        return std::make_pair(cv::Mat(), cv::Mat());
    }

    // 16bit的gray图像转换成8bit的gray图像（不影响最后拼接图像的位数，只被中间计算单应性矩阵H使用）
    cv::Mat gray_low_stop, gray_low_restart;
    low_energy_before_stop.convertTo(gray_low_stop, CV_8UC1, 1.0 / 256.0); // 1.0/256.0 缩放图像到[0,1]
    low_energy_after_restart.convertTo(gray_low_restart, CV_8UC1, 1.0 / 256.0);

    // 图像预处理，8bit进行（only low energy image）
    // clipLimit: 对比度限制阈值; tileGridSize: 均衡化网格的大小 
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(31, 31)); // 创建 CLAHE 对象

    // CLAHE灰度直方图预处理（only low energy image）
    cv::Mat clahe_image_low_stop;
    cv::Mat clahe_image_low_restart;

    clahe->apply(gray_low_stop, clahe_image_low_stop);
    clahe->apply(gray_low_restart, clahe_image_low_restart);

    // SIFT特征点检测和描述子提取（仅低能）
    auto [kp_low_before, des_low_before] = sift_feature_detector_and_descriptor(clahe_image_low_stop);
    auto [kp_low_after, des_low_after] = sift_feature_detector_and_descriptor(clahe_image_low_restart);

    // 特征点匹配（仅低能）
    auto matches_low = feature_matching(des_low_before, des_low_after);

    // 使用RANSAC计算单应矩阵（仅低能）
    auto [H_low, mask_low] = ransac_homography(kp_low_before, kp_low_after, matches_low);

    //输出调试信息
    if (!H_low.empty()) {
        // 拼接图像（低能）
        cv::Mat stitched_low = stitch_images(low_energy_after_restart, low_energy_before_stop, H_low);

        // 拼接图像（高能）[使用相同的单应矩阵H_low拼接高能图像]
        cv::Mat stitched_high = stitch_images(high_energy_after_restart, high_energy_before_stop, H_low);

        return { stitched_high, stitched_low };
    }
    else {
        std::cerr << "Not enough matches found to compute homography for low energy images" << std::endl;
        return { cv::Mat(), cv::Mat() };
    }
}

// 前后图像拼接
cv::Mat stitch_images_new(const cv::Mat& before_stop, const cv::Mat& after_restart) {
    // 判断输入图像状态
    if (before_stop.empty() || after_restart.empty()) {
        std::cerr << "One or both images are empty!" << std::endl;
        return cv::Mat();
    }

    if (before_stop.type() != CV_16UC1 || after_restart.type() != CV_16UC1) {
        std::cerr << "Error: Images are not 16-bit single channel (grayscale)!" << std::endl;
        return cv::Mat();
    }

    // 16-bit灰度图像转换为8-bit灰度图像
    cv::Mat gray_before, gray_after;
    before_stop.convertTo(gray_before, CV_8UC1, 1.0 / 256.0);
    after_restart.convertTo(gray_after, CV_8UC1, 1.0 / 256.0);

    // 图像预处理，使用CLAHE对比度限制自适应直方图均衡化
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(31, 31));
    cv::Mat clahe_before, clahe_after;
    clahe->apply(gray_before, clahe_before);
    clahe->apply(gray_after, clahe_after);

    // SIFT特征点检测和描述子提取
    auto [kp_before, des_before] = sift_feature_detector_and_descriptor(clahe_before);
    auto [kp_after, des_after] = sift_feature_detector_and_descriptor(clahe_after);

    // 特征点匹配
    auto matches = feature_matching(des_before, des_after);

    // 使用RANSAC计算单应矩阵
    auto [H, mask] = ransac_homography(kp_before, kp_after, matches);

    if (!H.empty()) {
        // 使用单应矩阵拼接图像
        cv::Mat stitched_image;
        cv::warpPerspective(after_restart, stitched_image, H,
            cv::Size(after_restart.cols + before_stop.cols, after_restart.rows));

        // 将前一个图像复制到拼接图像的左侧
        before_stop.copyTo(stitched_image(cv::Rect(0, 0, before_stop.cols, before_stop.rows)));

        return stitched_image;
    }
    else {
        std::cerr << "Not enough matches found to compute homography" << std::endl;
        return cv::Mat();
    }
}





// 该函数处理图像的缓冲和拼接操作。当满足一定条件时（重启后的图像宽度达到了停止前图像宽度的20%以上），
// 执行图像拼接操作，并返回拼接后的高能和低能图像。如果条件不满足，则返回空的图像对。
// ***how to use this function***
// 初始化缓冲区
// std::queue<cv::Mat> low_before_buffer;
// std::queue<cv::Mat> low_restart_buffer;
// int buffer_size = 10;  
//std::pair<cv::Mat, cv::Mat> bufferAndStitchImages(const cv::Mat& high_before,   // 高能停止前的Img
//    const cv::Mat& high_restart,  // 高能重启后的Img
//    const cv::Mat& low_before,    // 低能停止前的Img
//    const cv::Mat& low_restart,   // 低能重启后的Img
//    std::queue<cv::Mat>& low_before_buffer,  // 缓存低能停止前Img
//    std::queue<cv::Mat>& low_restart_buffer,  // 缓存低能重启后Img
//    int buffer_size) {                        // 缓冲区的最大容量
//    // 添加图像到缓冲区
//    if (low_before_buffer.size() >= buffer_size) {
//        low_before_buffer.pop();  // 如果缓冲区满了，移除最旧的图像
//    }
//    if (low_restart_buffer.size() >= buffer_size) {
//        low_restart_buffer.pop();  // 如果缓冲区满了，移除最旧的图像
//    }
//    low_before_buffer.push(low_before);
//    low_restart_buffer.push(low_restart);
//
//    // 检查重启后的图像是否达到了停止前图像宽度的20%
//    if (!low_before_buffer.empty() && !low_restart_buffer.empty()) {
//        cv::Mat low_before = low_before_buffer.back();
//        cv::Mat low_restart = low_restart_buffer.back();
//
//        if (low_restart.cols >= low_before.cols * 0.2) {
//            // 满足条件，进行图像拼接
//            return stitch_dual_energy_images(high_before, high_restart, low_before, low_restart);
//        }
//    }
//
//    // 如果条件不满足，返回空的 Mat 对象对
//    return std::make_pair(cv::Mat(), cv::Mat());
//}





std::pair<cv::Mat, cv::Mat> bufferAndStitchImages(const cv::Mat& high_before,   // 高能停止前的Img
    const cv::Mat& high_restart,  // 高能重启后的Img
    const cv::Mat& low_before,    // 低能停止前的Img
    const cv::Mat& low_restart)  // 低能重启后的Img

{

    return stitch_dual_energy_images(high_before, high_restart, low_before, low_restart);

}

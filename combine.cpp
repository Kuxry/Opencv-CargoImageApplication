//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//int main() {
//    // 读取两张图片
//    cv::Mat img1 = cv::imread("0607.png", cv::IMREAD_COLOR);
//    cv::Mat img2 = cv::imread("0608.png", cv::IMREAD_COLOR);
//
//    // 检查图片是否成功加载
//    if (img1.empty()) {
//        std::cerr << "无法读取图片 0607.png" << std::endl;
//        return -1;
//    }
//    if (img2.empty()) {
//        std::cerr << "无法读取图片 0608.png" << std::endl;
//        return -1;
//    }
//
//    // 逆时针旋转90度每张图片
//    cv::Mat rotatedImg1, rotatedImg2;
//    cv::rotate(img1, rotatedImg1, cv::ROTATE_90_COUNTERCLOCKWISE);
//    cv::rotate(img2, rotatedImg2, cv::ROTATE_90_COUNTERCLOCKWISE);
//
//    // 输出旋转后图片的宽度和高度
//    std::cout << "旋转后的图片1宽度: " << rotatedImg1.cols << std::endl;
//    std::cout << "旋转后的图片1高度: " << rotatedImg1.rows << std::endl;
//    std::cout << "旋转后的图片2宽度: " << rotatedImg2.cols << std::endl;
//    std::cout << "旋转后的图片2高度: " << rotatedImg2.rows << std::endl;
//
//    // 检查两张图片的高度是否相同（对于水平拼接）或者宽度是否相同（对于垂直拼接）
//    if (rotatedImg1.rows != rotatedImg2.rows && rotatedImg1.cols != rotatedImg2.cols) {
//        std::cerr << "图片尺寸不一致，无法拼接" << std::endl;
//        return -1;
//    }
//
//    // 选择拼接方式
//    bool verticalConcat = false;  // false: 水平拼接; true: 垂直拼接
//
//    cv::Mat result;
//    if (verticalConcat) {
//        // 垂直拼接图片
//        cv::vconcat(rotatedImg1, rotatedImg2, result);
//    }
//    else {
//        // 水平拼接图片
//        cv::hconcat(rotatedImg1, rotatedImg2, result);
//    }
//
//    // 输出拼接后图片的宽度和高度
//    std::cout << "拼接后的图片宽度: " << result.cols << std::endl;
//    std::cout << "拼接后的图片高度: " << result.rows << std::endl;
//
//    // 保存拼接后的图片
//    std::string outputFileName = "combined_image.png";
//    cv::imwrite(outputFileName, result);
//
//    // 显示拼接后的图片
//    cv::imshow("拼接后的图片", result);
//
//    // 等待用户按键
//    cv::waitKey(0);
//
//    return 0;
//}

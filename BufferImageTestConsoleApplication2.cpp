#include "txr_CBpDetectorImgBuf.h"

#include "stitching.h"
#include <string>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <thread>
#include <atomic>
#include <queue>
#include <opencv2/highgui.hpp> // Added for imshow and waitKey

namespace fs = std::filesystem;

// Keeping loadAndSplitImage function unchanged
std::unique_ptr<txr_CBpDetectorImgBuf> loadAndSplitImage(const std::string& filename, int width, int height) {
    int fullWidth = width;
    int halfWidth = fullWidth;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return nullptr;
    }

    std::vector<uint16_t> fullImageData(fullWidth * height);
    file.read(reinterpret_cast<char*>(fullImageData.data()), fullImageData.size() * sizeof(uint16_t));

    if (!file) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        return nullptr;
    }

    auto detector = std::make_unique<txr_CBpDetectorImgBuf>(16, halfWidth, height);
    for (int y = 0; y < height; ++y) {
        detector->lowEnergyData.insert(detector->lowEnergyData.end(),
            fullImageData.begin() + y * fullWidth,
            fullImageData.begin() + y * fullWidth + halfWidth);


    }

    return detector;
}
// Convert to txr_CBpDetectorImgBuf
std::unique_ptr<txr_CBpDetectorImgBuf> matToTxrCBpDetectorImgBuf(const cv::Mat& stitched_high, const cv::Mat& stitched_low, int width) {
    int halfWidth = width / 2;
    int height = stitched_high.rows;
    auto detector = std::make_unique<txr_CBpDetectorImgBuf>(16, halfWidth, height);

    // Assuming the stitched images are vertically concatenated slices
    for (int y = 0; y < stitched_high.rows; ++y) {
        const uint16_t* highRow = stitched_high.ptr<uint16_t>(y);
        const uint16_t* lowRow = stitched_low.ptr<uint16_t>(y);

        detector->highEnergyData.insert(detector->highEnergyData.end(), highRow, highRow + halfWidth);
        detector->lowEnergyData.insert(detector->lowEnergyData.end(), lowRow, lowRow + halfWidth);
    }

    return detector;
}


// Stitch all slices up to the pause point and return the stitched images
std::pair<cv::Mat, cv::Mat> stitchFirstBeforeSlices(const std::vector<txr_CBpDetectorImgBuf*>& txr_imgs, int width, int height) {
    if (!txr_imgs.empty()) {
        std::vector<cv::Mat> pngImages;

        for (auto* txr_img : txr_imgs) {
            if (!txr_img) {
                continue;
            }

            cv::Mat image_temp(height, width, CV_16UC1, txr_img->lowEnergyData.data());
            pngImages.push_back(image_temp.clone());
        }

        if (!pngImages.empty()) {
            cv::Mat stitchedImage;
            cv::vconcat(pngImages, stitchedImage);

            std::cout << "Stitched all slices in memory." << std::endl;
            return { stitchedImage, stitchedImage };
        }
        else {
            std::cout << "Insufficient image data for stitching all slices" << std::endl;
        }
    }
    else {
        std::cout << "No image data provided for stitching all slices" << std::endl;
    }

    return { cv::Mat(), cv::Mat() };
}

// Stitch and return the last 10 slices
std::pair<cv::Mat, cv::Mat> stitchRecent10Slices(const std::deque<txr_CBpDetectorImgBuf*>& recent_txr_imgs, int width, int height) {
    if (!recent_txr_imgs.empty()) {
        std::vector<cv::Mat> pngImages;

        for (auto* txr_img : recent_txr_imgs) {
            if (!txr_img) {
                continue;
            }

            cv::Mat image_temp(height, width, CV_16UC1, txr_img->lowEnergyData.data());
            pngImages.push_back(image_temp.clone());
        }

        if (!pngImages.empty()) {
            cv::Mat stitchedImage;
            cv::vconcat(pngImages, stitchedImage);

            std::cout << "Stitched the last 10 slices in memory." << std::endl;
            return { stitchedImage, stitchedImage };
        }
        else {
            std::cout << "Insufficient image data for stitching last 10 slices" << std::endl;
        }
    }
    else {
        std::cout << "No recent image data provided for stitching" << std::endl;
    }

    return { cv::Mat(), cv::Mat() };
}

int main() {
    int width = 2304;
    int height = 24;


    std::vector<std::string> fileNames;
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.is_regular_file() && entry.path().extension() == ".raw") {
            fileNames.push_back(entry.path().string());
        }
    }

    std::vector<std::unique_ptr<txr_CBpDetectorImgBuf>> detectors;
    std::deque<txr_CBpDetectorImgBuf*> recent_detectors;
    std::atomic<bool> stop_flag(false);

    std::thread loading_thread([&]() {
        int fileIndex = 0;
        while (!stop_flag) {
            if (fileIndex < fileNames.size()) {
                auto detector = loadAndSplitImage(fileNames[fileIndex], width, height);
                if (detector) {
                    detectors.push_back(std::move(detector));
                    if (detectors.back()) {
                        recent_detectors.push_back(detectors.back().get());

                        if (recent_detectors.size() > 10) {
                            recent_detectors.pop_front();
                        }
                    }
                    std::cout << "Loading a new image every second: " << fileNames[fileIndex] << std::endl;
                }
                else {
                    std::cerr << "Failed to process file: " << fileNames[fileIndex] << std::endl;
                }
                fileIndex++;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        });

    std::cout << "Press 'p' to pause and stitch images..." << std::endl;
    while (true) {
        if (std::cin.get() == 'p') {
            stop_flag = true;
            loading_thread.join();

            std::vector<txr_CBpDetectorImgBuf*> all_slices;
            for (const auto& detector : detectors) {
                all_slices.push_back(detector.get());
            }
            auto [low_stitched_before, high_stitched_before] = stitchFirstBeforeSlices(all_slices, width, height);

            auto [low_stitched_recent, high_stitched_recent] = stitchRecent10Slices(recent_detectors, width, height);

            if (low_stitched_before.empty() || low_stitched_recent.empty()) {
                std::cerr << "Error stitching images in memory" << std::endl;
                return -1;
            }

            auto stitched = bufferAndStitchImages(low_stitched_before, low_stitched_recent, low_stitched_before, low_stitched_recent);

            // Convert the stitched images back to txr_CBpDetectorImgBuf
            auto detector_from_stitched = matToTxrCBpDetectorImgBuf(
                stitched.first,
                stitched.second,
                width);


            std::cout << "Final images stitched in memory." << std::endl;

            cv::imshow("Stitched Image", stitched.first);
            cv::waitKey(0);

            break;
        }
    }

    return 0;
}

// Copyright@ biomed ssSEM Lab
// Contact@ Hongyu Ge, horacekem@163.com
// Time@ 2021.04.08

// This code provides a benchmark of several feature points extraction algorithms implemented by OpenCV.
// To test all algorithms, you have to ensure that the following options were set correct when installing OpenCV.
// >>> -DOPENCV_EXTRA_MODULES_PATH={prefix}/opencv_contrib-master/modules
// >>> -DWITH_CUDA=ON
// >>> -DOPENCV_ENABLE_NONFREE=ON

// All codes were tested on Ubuntu18.04_x86_64, OpenCV Version= 4.5.2
// Note that the APIs in different versions of OpenCV are not identical, please modify the codes referring to
// the official document[https://opencv.org/] if your version is not 4.5.2.

#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "featuresExtractor.h"
#include "featuresMatching.h"
#include "runParams.h"
#include "inlierEstimation.h"
#include "jsonIO.h"

#define MICROSECONDS_TO_SECONDS 1e-6
using namespace cv;

void printExtractorResult(std::vector<KeyPoint>& _keyPoints, double _time_taken){
    std::cout << "\tThe run time is: " << _time_taken << "s" << std::endl;
    int num = _keyPoints.size();
    std::cout << "\tThe number of keyPoints is: " << num << std::endl;
}

void runExtractorTests(Mat& _img){
    std::vector<KeyPoint> keyPoints;

    if (run_Extractor_sift) {
        auto siftStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = siftDetector(_img);
        auto siftEndTime = std::chrono::high_resolution_clock::now();
        double siftTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(siftEndTime - siftStartTime).count();
        siftTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "SIFT result:" << std::endl;
        printExtractorResult(keyPoints, siftTimeTaken);
    }

    if (run_Extractor_surf) {
        auto surfStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = surfDetector(_img);
        auto surfEndTime = std::chrono::high_resolution_clock::now();
        double surfTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(surfEndTime - surfStartTime).count();
        surfTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "SURF result:" << std::endl;
        printExtractorResult(keyPoints, surfTimeTaken);
    }

    if (run_Extractor_orb) {
        auto orbStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = orbDetector(_img);
        auto orbEndTime = std::chrono::high_resolution_clock::now();
        double orbTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(orbEndTime - orbStartTime).count();
        orbTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "ORB result:" << std::endl;
        printExtractorResult(keyPoints, orbTimeTaken);
    }

    if (run_Extractor_surf_cuda) {
        auto surfCudaStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = surfCudaDetector(_img);
        auto surfCudaEndTime = std::chrono::high_resolution_clock::now();
        double surfCudaTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(surfCudaEndTime - surfCudaStartTime).count();
        surfCudaTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "SURF_CUDA result:" << std::endl;
        printExtractorResult(keyPoints, surfCudaTimeTaken);
    }

    if (run_Extractor_orb_cuda) {
        auto orbCudaStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = orbCudaDetector(_img);
        auto orbCudaEndTime = std::chrono::high_resolution_clock::now();
        double orbCudaTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(orbCudaEndTime - orbCudaStartTime).count();
        orbCudaTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "ORB_CUDA result:" << std::endl;
        printExtractorResult(keyPoints, orbCudaTimeTaken);
    }

    if (run_Extractor_akaze) {
        auto akazeStartTime = std::chrono::high_resolution_clock::now();
        keyPoints = akazeDetector(_img);
        auto akazeEndTime = std::chrono::high_resolution_clock::now();
        double akazeTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(akazeEndTime - akazeStartTime).count();
        akazeTimeTaken *= MICROSECONDS_TO_SECONDS;
        std::cout << "AKAZE result:" << std::endl;
        printExtractorResult(keyPoints, akazeTimeTaken);
    }
}

void runMatchingTests(Mat& _img1, Mat& _img2){
    matchedFeatures matched_features;
    String save_path = result_path;

    if(run_Matching_sift) {
        String sift_folder = save_path + "/1-sift";
        const char *path = sift_folder.data();
        mkdir(path, 00700);

        auto siftStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running SIFT testing..." << std::endl;
        matched_features = siftMatching(_img1, _img2);
        auto siftEndTime = std::chrono::high_resolution_clock::now();
        double siftTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(siftEndTime - siftStartTime).count();
        siftTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = sift_folder + "/result.json";
        String save_matches_path = sift_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, siftTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(sift_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(sift_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(sift_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << sift_folder << std::endl;
        }
    }

    if(run_Matching_surf) {
        String surf_folder = save_path + "/2-surf";
        const char *path = surf_folder.data();
        mkdir(path, 00700);

        auto surfStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running SURF testing..." << std::endl;
        matched_features = surfMatching(_img1, _img2);
        auto surfEndTime = std::chrono::high_resolution_clock::now();
        double surfTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(surfEndTime - surfStartTime).count();
        surfTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = surf_folder + "/result.json";
        String save_matches_path = surf_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, surfTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(surf_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(surf_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(surf_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << surf_folder << std::endl;
        }
    }

    if(run_Matching_orb) {
        String orb_folder = save_path + "/3-orb";
        const char *path = orb_folder.data();
        mkdir(path, 00700);
        auto orbStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running ORB testing..." << std::endl;
        matched_features = orbMatching(_img1, _img2);
        auto orbEndTime = std::chrono::high_resolution_clock::now();
        double orbTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(orbEndTime - orbStartTime).count();
        orbTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = orb_folder + "/result.json";
        String save_matches_path = orb_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, orbTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(orb_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(orb_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(orb_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << orb_folder << std::endl;
        }
    }

    if(run_Matching_orb_cuda) {
        String orb_cuda_folder = save_path + "/4-orb_cuda";
        const char *path = orb_cuda_folder.data();
        mkdir(path, 00700);

        auto orbCudaStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running ORB_CUDA testing..." << std::endl;
        matched_features = orbCudaMatching(_img1, _img2);
        auto orbCudaEndTime = std::chrono::high_resolution_clock::now();
        double orbCudaTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(orbCudaEndTime - orbCudaStartTime).count();
        orbCudaTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = orb_cuda_folder + "/result.json";
        String save_matches_path = orb_cuda_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, orbCudaTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(orb_cuda_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(orb_cuda_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(orb_cuda_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << orb_cuda_folder << std::endl;
        }
    }

    if(run_Matching_surf_cuda_gpu) {
        String surf_cuda_folder = save_path + "/5-surf_cuda";
        const char *path = surf_cuda_folder.data();
        mkdir(path, 00700);

        auto surfCudaGPUStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running SURF_CUDA_GPU testing..." << std::endl;
        matched_features = surfCudaGPUMatching(_img1, _img2);
        auto surfCudaGPUEndTime = std::chrono::high_resolution_clock::now();
        double surfCudaGPUTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(surfCudaGPUEndTime - surfCudaGPUStartTime).count();
        surfCudaGPUTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = surf_cuda_folder + "/result.json";
        String save_matches_path = surf_cuda_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, surfCudaGPUTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(surf_cuda_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(surf_cuda_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(surf_cuda_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << surf_cuda_folder << std::endl;
        }
    }

    if(run_Matching_akaze) {
        String akaze_folder = save_path + "/6-akaze";
        const char *path = akaze_folder.data();
        mkdir(path, 00700);

        auto akazeStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Running AKAZE testing..." << std::endl;
        matched_features = akazeMatching(_img1, _img2);
        auto akazeEndTime = std::chrono::high_resolution_clock::now();
        double akazeTimeTaken = std::chrono::duration_cast<std::chrono::microseconds>(akazeEndTime - akazeStartTime).count();
        akazeTimeTaken *= MICROSECONDS_TO_SECONDS;

        inliersInfo inlier_info = inlierEstiWithoutHomo(matched_features);
        String save_result_path = akaze_folder + "/result.json";
        String save_matches_path = akaze_folder + "/matches.json";
        writeRunningResult(save_result_path, matched_features, akazeTimeTaken, inlier_info);
        writeFilteredMatches(save_matches_path, inlier_info);
        std::cout << "\tResult saved at " << save_result_path << std::endl;
        if(save_image){
            Mat img_matches;
            drawMatches(_img1, matched_features.extractedFeatures1.keyPoints, _img2, matched_features.extractedFeatures2.keyPoints, matched_features.matches.pair_matches, img_matches);
            imwrite(akaze_folder + "/matching.png", img_matches);
            img_matches.release();
            Mat img1_kps;
            drawKeypoints(_img1, matched_features.extractedFeatures1.keyPoints, img1_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(akaze_folder + "/img1_kps.png", img1_kps);
            img1_kps.release();
            Mat img2_kps;
            drawKeypoints(_img2, matched_features.extractedFeatures2.keyPoints, img2_kps, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imwrite(akaze_folder + "/img2_kps.png", img2_kps);
            img2_kps.release();
            std::cout << "\tImages saved at " << akaze_folder << std::endl;
        }
    }
}

int main() {
    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    // Load image.
    String img1_path = "../imgs/2048/Tile_r2-c2_S_001_920258840.tif";
    String img2_path = "../imgs/2048/Tile_r2-c3_S_001_920258840.tif";
    Mat img1, img2;
    img1 = imread(img1_path, cv::IMREAD_GRAYSCALE);
    img2 = imread(img2_path, cv::IMREAD_GRAYSCALE);
    std::cout << "Image Dimensions: \n1-" << img1.size << "\n2-" << img2.size << std::endl;
    //Run all tests.
//    runExtractorTests(img1);
    runMatchingTests(img1, img2);
    return 0;
}

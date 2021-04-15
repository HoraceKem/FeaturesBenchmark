// @Time : 2021/4/9 下午3:10
// @Author : Horace.Kem
// @File: featuresMatching.cpp
// @Software: CLion
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <iostream>
#include "featuresExtractor.h"
#include "featuresParams.h"

using namespace cv;

matchesAll flannBasedMatcher(features& _fea1, features& _fea2){
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(_fea1.descriptors, _fea2.descriptors, knn_matches, 2);

    const float  ratio_threshold = 0.7f;
    matchesAll good_matches;
    for (auto & knn_match : knn_matches){
        if (knn_match[0].distance < ratio_threshold * knn_match[1].distance){
            good_matches.pair_matches.push_back(knn_match[0]);
            good_matches.matches1.push_back(_fea1.keyPoints[knn_match[0].queryIdx]);
            good_matches.matches2.push_back(_fea2.keyPoints[knn_match[0].trainIdx]);
        }
    }
    return good_matches;
}

matchesAll desConvertFlannBasedMatcher(features& _fea1, features& _fea2){
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    Mat convertedDes1, convertedDes2;
    _fea1.descriptors.convertTo(convertedDes1, CV_32F);
    _fea2.descriptors.convertTo(convertedDes2, CV_32F);
    matcher->knnMatch(convertedDes1, convertedDes2, knn_matches, 2);

    const float  ratio_threshold = 0.7f;
    matchesAll good_matches;
    for (auto & knn_match : knn_matches){
        if (knn_match[0].distance < ratio_threshold * knn_match[1].distance){
            good_matches.pair_matches.push_back(knn_match[0]);
            good_matches.matches1.push_back(_fea1.keyPoints[knn_match[0].queryIdx]);
            good_matches.matches2.push_back(_fea2.keyPoints[knn_match[0].trainIdx]);
        }
    }
    return good_matches;
}

matchesAll BFMatcher(features& _fea1, features& _fea2){
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(_fea1.descriptors, _fea2.descriptors, knn_matches, 2);

    const float  ratio_threshold = 0.7f;
    matchesAll good_matches;
    for (auto & knn_match : knn_matches){
        if (knn_match[0].distance < ratio_threshold * knn_match[1].distance){
            good_matches.pair_matches.push_back(knn_match[0]);
            good_matches.matches1.push_back(_fea1.keyPoints[knn_match[0].queryIdx]);
            good_matches.matches2.push_back(_fea2.keyPoints[knn_match[0].trainIdx]);
        }
    }
    return good_matches;
}

matchesAll GPUMatcher(featuresGPU& _fea1, featuresGPU& _fea2){
    cuda::SURF_CUDA surf;
    Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(_fea1.descriptorsGPU, _fea2.descriptorsGPU, knn_matches, 2);

    const float  ratio_threshold = 0.7f;
    matchesAll good_matches;
    for (auto & knn_match : knn_matches){
        if (knn_match[0].distance < ratio_threshold * knn_match[1].distance){
            good_matches.pair_matches.push_back(knn_match[0]);
            good_matches.matches1.push_back(_fea1.keyPointsCPU[knn_match[0].queryIdx]);
            good_matches.matches2.push_back(_fea2.keyPointsCPU[knn_match[0].trainIdx]);
        }
    }
    return good_matches;
}

matchedFeatures siftMatching(Mat& _img1, Mat& _img2){
    features siftFeatures1 = siftComputer(_img1);
    features siftFeatures2 = siftComputer(_img2);
    matchedFeatures siftMatchedFeatures;
    siftMatchedFeatures.matches = flannBasedMatcher(siftFeatures1, siftFeatures2);
    siftMatchedFeatures.extractedFeatures1 = siftFeatures1;
    siftMatchedFeatures.extractedFeatures2 = siftFeatures2;
    return siftMatchedFeatures;
}

matchedFeatures surfMatching(Mat& _img1, Mat& _img2){
    features surfFeatures1 = surfComputer(_img1);
    features surfFeatures2 = surfComputer(_img2);
    matchedFeatures surfMatchedFeatures;
    surfMatchedFeatures.matches = flannBasedMatcher(surfFeatures1, surfFeatures2);
    surfMatchedFeatures.extractedFeatures1 = surfFeatures1;
    surfMatchedFeatures.extractedFeatures2 = surfFeatures2;
    return surfMatchedFeatures;
}

matchedFeatures orbMatching(Mat& _img1, Mat& _img2){
    features orbFeatures1 = orbComputer(_img1);
    features orbFeatures2 = orbComputer(_img2);
    matchedFeatures orbMatchedFeatures;
    orbMatchedFeatures.matches = orb_matcher(orbFeatures1, orbFeatures2);
    orbMatchedFeatures.extractedFeatures1 = orbFeatures1;
    orbMatchedFeatures.extractedFeatures2 = orbFeatures2;
    return orbMatchedFeatures;
}

matchedFeatures orbCudaMatching(Mat& _img1, Mat& _img2){
    features orbCudaFeatures1 = orbCudaComputer(_img1);
    features orbCudaFeatures2 = orbCudaComputer(_img2);
    matchedFeatures orbCudaMatchedFeatures;
    orbCudaMatchedFeatures.matches = orb_cuda_matcher(orbCudaFeatures1, orbCudaFeatures2);
    orbCudaMatchedFeatures.extractedFeatures1 = orbCudaFeatures1;
    orbCudaMatchedFeatures.extractedFeatures2 = orbCudaFeatures2;
    return orbCudaMatchedFeatures;
}

matchedFeatures surfCudaGPUMatching(Mat& _img1, Mat& _img2){
    featuresGPU surfCudaGPUFeatures1 = surfCudaComputerGPU(_img1);
    featuresGPU surfCudaGPUFeatures2 = surfCudaComputerGPU(_img2);
    features surfCudaFeatures1;
    features surfCudaFeatures2;
    surfCudaFeatures1.keyPoints = surfCudaGPUFeatures1.keyPointsCPU;
    surfCudaFeatures1.descriptors = Mat(surfCudaGPUFeatures1.descriptorsCPU);
    surfCudaFeatures2.keyPoints = surfCudaGPUFeatures2.keyPointsCPU;
    surfCudaFeatures2.descriptors = Mat(surfCudaGPUFeatures2.descriptorsCPU);

    matchedFeatures surfCudaMatchedFeatures;
    surfCudaMatchedFeatures.matches = GPUMatcher(surfCudaGPUFeatures1, surfCudaGPUFeatures2);
    surfCudaMatchedFeatures.extractedFeatures1 = surfCudaFeatures1;
    surfCudaMatchedFeatures.extractedFeatures2 = surfCudaFeatures2;
    return surfCudaMatchedFeatures;
}

matchedFeatures akazeMatching(Mat& _img1, Mat& _img2){
    features akazeFeatures1 = akazeComputer(_img1);
    features akazeFeatures2 = akazeComputer(_img2);
    matchedFeatures akazeMatchedFeatures;
    akazeMatchedFeatures.matches = akaze_matcher(akazeFeatures1, akazeFeatures2);
    akazeMatchedFeatures.extractedFeatures1 = akazeFeatures1;
    akazeMatchedFeatures.extractedFeatures2 = akazeFeatures2;
    return akazeMatchedFeatures;
}
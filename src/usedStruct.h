//
// Created by sem-lab on 2021/4/11.
//
#ifndef FEATURESBENCHMARK_USEDSTRUCT_H
#define FEATURESBENCHMARK_USEDSTRUCT_H

#include <opencv2/core/cuda.hpp>
using namespace cv;

typedef struct{
    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
}features;

typedef struct{
    cuda::GpuMat keyPointsGPU;
    cuda::GpuMat descriptorsGPU;
    std::vector<KeyPoint> keyPointsCPU;
    std::vector<float> descriptorsCPU;
}featuresGPU;

typedef struct{
    std::vector<DMatch> pair_matches;
    std::vector<KeyPoint> matches1;
    std::vector<KeyPoint> matches2;
}matchesAll;

typedef struct{
    features extractedFeatures1;
    features extractedFeatures2;
    matchesAll matches;
}matchedFeatures;

typedef struct{
    int inliers;
    Mat homography;
    double inliers_ratio;
    std::vector<std::vector<Point2f>> filtered_good_matches;
}inliersInfo;

#endif //FEATURESBENCHMARK_USEDSTRUCT_H


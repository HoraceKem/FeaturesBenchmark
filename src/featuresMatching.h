//
// Created by sem-lab on 2021/4/11.
//
#ifndef FEATURESBENCHMARK_FEATURESMATCHING_H
#define FEATURESBENCHMARK_FEATURESMATCHING_H

#include "usedStruct.h"
using namespace cv;

matchesAll flannBasedMatcher(features& _fea1, features& _fea2);
matchesAll desConvertFlannBasedMatcher(features& _fea1, features& _fea2);
matchesAll BFMatcher(features& _fea1, features& _fea2);
matchesAll GPUMatcher(featuresGPU& _fea1, featuresGPU& _fea2);

matchedFeatures siftMatching(Mat& _img1, Mat& _img2);
matchedFeatures surfMatching(Mat& _img1, Mat& _img2);
matchedFeatures orbMatching(Mat& _img1, Mat& _img2);
matchedFeatures orbCudaMatching(Mat& _img1, Mat& _img2);
matchedFeatures surfCudaGPUMatching(Mat& _img1, Mat& _img2);
matchedFeatures akazeMatching(Mat& _img1, Mat& _img2);

#endif //FEATURESBENCHMARK_FEATURESMATCHING_H


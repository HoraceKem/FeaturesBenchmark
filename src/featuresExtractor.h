//
// Created by sem-lab on 2021/4/9.
//
#ifndef FEATURESBENCHMARK_FEATURESEXTRACTOR_H
#define FEATURESBENCHMARK_FEATURESEXTRACTOR_H

#include "usedStruct.h"
using namespace cv;

std::vector<KeyPoint> siftDetector(Mat& _img);
std::vector<KeyPoint> surfDetector(Mat& _img);
std::vector<KeyPoint> orbDetector(Mat& _img);
std::vector<KeyPoint> surfCudaDetector(Mat& _img);
std::vector<KeyPoint> orbCudaDetector(Mat& _img);
std::vector<KeyPoint> akazeDetector(Mat& _img);

features siftComputer(Mat& _img);
features surfComputer(Mat& _img);
features orbComputer(Mat& _img);
features orbCudaComputer(Mat& _img);
featuresGPU surfCudaComputerGPU(Mat& _img);
features akazeComputer(Mat& _img);

#endif //FEATURESBENCHMARK_FEATURESEXTRACTOR_H

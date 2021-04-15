// @Time : 2021/4/9 上午11:32
// @Author : Horace.Kem
// @File: featuresdetector.cpp
// @Software: CLion
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "featuresExtractor.h"
#include "featuresParams.h"

using namespace cv;

std::vector<KeyPoint> siftDetector(Mat& _img){
    Ptr<SIFT> detectorSift = SIFT::create(sift_nfeatures, sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma);
    std::vector<KeyPoint> keyPoints;

    detectorSift->detect(_img, keyPoints);
    return keyPoints;
}

std::vector<KeyPoint> surfDetector(Mat& _img){
    Ptr<xfeatures2d::SURF> detectorSurf = xfeatures2d::SURF::create(surf_hessianThreshold, surf_nOctaves, surf_nOctaveLayers, surf_extended, surf_upright);
    std::vector<KeyPoint> keyPoints;

    detectorSurf->detect(_img, keyPoints);
    return keyPoints;
}

std::vector<KeyPoint> orbDetector(Mat& _img){
    Ptr<ORB> detectorOrb = ORB::create(orb_nfeatures, orb_scaleFactor, orb_nlevels, orb_edgeThreshold, orb_firstLevel, orb_WTA_K, orb_scoreType, orb_patchSize, orb_fastThreshold);
    std::vector<KeyPoint> keyPoints;

    detectorOrb->detect(_img, keyPoints);
    return keyPoints;
}

std::vector<KeyPoint> surfCudaDetector(Mat& _img){
    Ptr<cuda::SURF_CUDA> detectorSurfCuda = cuda::SURF_CUDA::create(surf_cuda_hessianThreshold, surf_cuda_nOctaves, surf_cuda_nOctaveLayers, surf_cuda_extended, surf_cuda_keypointsRatio, surf_cuda_upright);
    cuda::GpuMat _img_gpu;
    _img_gpu.upload(_img);
    cuda::GpuMat keyPoints_gpu;
    std::vector<KeyPoint> keyPoints;

    detectorSurfCuda->detect(_img_gpu, cuda::GpuMat(), keyPoints_gpu);
    detectorSurfCuda->downloadKeypoints(keyPoints_gpu, keyPoints);
    return keyPoints;
}

std::vector<KeyPoint> orbCudaDetector(Mat& _img){
    Ptr<cuda::ORB> detectorOrbCuda = cuda::ORB::create(orb_cuda_nfeatures, orb_cuda_scaleFactor, orb_cuda_nlevels, orb_cuda_edgeThreshold, orb_cuda_firstLevel, orb_cuda_WTA_K, orb_cuda_scoreType, orb_cuda_patchSize, orb_cuda_fastThreshold, orb_cuda_blurForDescriptor);
    cuda::GpuMat _img_gpu;
    _img_gpu.upload(_img);
    std::vector<KeyPoint> keyPoints;

    detectorOrbCuda->detect(_img_gpu, keyPoints);
    return keyPoints;
}

std::vector<KeyPoint> akazeDetector(Mat& _img){
    Ptr<AKAZE> detectorAkaze = AKAZE::create(akaze_descriptor_type, akaze_descriptor_size, akaze_descriptor_channels, akaze_threshold, akaze_nOctaves, akaze_nOctaveLayers, akaze_diffusivity);
    std::vector<KeyPoint> keyPoints;

    detectorAkaze->detect(_img, keyPoints);
    return keyPoints;
}

features siftComputer(Mat& _img){
    Ptr<SIFT> computerSift = SIFT::create(sift_nfeatures, sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma);
    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
    features siftFeatures;

    computerSift->detectAndCompute(_img, noArray(), keyPoints, descriptors);
    siftFeatures.keyPoints = keyPoints;
    siftFeatures.descriptors = descriptors;
    return siftFeatures;
}

features surfComputer(Mat& _img){
    Ptr<xfeatures2d::SURF> computerSurf = xfeatures2d::SURF::create(surf_hessianThreshold, surf_nOctaves, surf_nOctaveLayers, surf_extended, surf_upright);
    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
    features surfFeatures;

    computerSurf->detectAndCompute(_img, noArray(), keyPoints, descriptors);
    surfFeatures.keyPoints = keyPoints;
    surfFeatures.descriptors = descriptors;
    return surfFeatures;
}

features orbComputer(Mat& _img){
    Ptr<ORB> computerOrb = ORB::create(orb_nfeatures, orb_scaleFactor, orb_nlevels, orb_edgeThreshold, orb_firstLevel, orb_WTA_K, orb_scoreType, orb_patchSize, orb_fastThreshold);
    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
    features orbFeatures;

    computerOrb->detectAndCompute(_img, noArray(), keyPoints, descriptors);
    orbFeatures.keyPoints = keyPoints;
    orbFeatures.descriptors = descriptors;
    return orbFeatures;
}

featuresGPU surfCudaComputerGPU(Mat& _img){
    Ptr<cuda::SURF_CUDA> computerSurfCuda = cuda::SURF_CUDA::create(surf_cuda_hessianThreshold, surf_cuda_nOctaves, surf_cuda_nOctaveLayers, surf_cuda_extended, surf_cuda_keypointsRatio, surf_cuda_upright);
    cuda::GpuMat _img_gpu;
    _img_gpu.upload(_img);
    cuda::GpuMat keyPoints_gpu;
    cuda::GpuMat descriptors_gpu;
    featuresGPU surfCudaFeaturesGPU;

    computerSurfCuda->detectWithDescriptors(_img_gpu, cuda::GpuMat(), keyPoints_gpu, descriptors_gpu);
    surfCudaFeaturesGPU.keyPointsGPU = keyPoints_gpu;
    surfCudaFeaturesGPU.descriptorsGPU = descriptors_gpu;
    computerSurfCuda->downloadKeypoints(keyPoints_gpu, surfCudaFeaturesGPU.keyPointsCPU);
    computerSurfCuda->downloadDescriptors(descriptors_gpu, surfCudaFeaturesGPU.descriptorsCPU);
    return surfCudaFeaturesGPU;
}

features orbCudaComputer(Mat& _img){
    Ptr<cuda::ORB> computerOrbCuda = cuda::ORB::create(orb_cuda_nfeatures, orb_cuda_scaleFactor, orb_cuda_nlevels, orb_cuda_edgeThreshold, orb_cuda_firstLevel, orb_cuda_WTA_K, orb_cuda_scoreType, orb_cuda_patchSize, orb_cuda_fastThreshold, orb_cuda_blurForDescriptor);
    cuda::GpuMat _img_gpu;
    _img_gpu.upload(_img);
    std::vector<KeyPoint> keyPoints;
    cuda::GpuMat descriptorsGPU;
    Mat descriptors;
    features orbCudaFeatures;

    computerOrbCuda->detectAndCompute(_img_gpu, noArray(), keyPoints, descriptorsGPU);
    descriptorsGPU.download(descriptors);
    orbCudaFeatures.keyPoints = keyPoints;
    orbCudaFeatures.descriptors = descriptors;
    return orbCudaFeatures;
}

features akazeComputer(Mat& _img){
    Ptr<AKAZE> computerAkaze = AKAZE::create(akaze_descriptor_type, akaze_descriptor_size, akaze_descriptor_channels, akaze_threshold, akaze_nOctaves, akaze_nOctaveLayers, akaze_diffusivity);
    std::vector<KeyPoint> keyPoints;
    Mat descriptors;
    features akazeFeatures;

    computerAkaze->detectAndCompute(_img, noArray(), keyPoints, descriptors);
    akazeFeatures.keyPoints = keyPoints;
    akazeFeatures.descriptors = descriptors;
    return akazeFeatures;
}

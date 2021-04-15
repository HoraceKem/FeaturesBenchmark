//
// Created by sem-lab on 2021/4/12.
//
#ifndef FEATURESBENCHMARK_FEATURESPARAMS_H
#define FEATURESBENCHMARK_FEATURESPARAMS_H

#define sift_nfeatures  0
#define sift_nOctaveLayers 3
#define sift_contrastThreshold 0.04
#define sift_edgeThreshold 10
#define sift_sigma 1.6

#define surf_hessianThreshold 100
#define surf_nOctaves 4
#define surf_nOctaveLayers 3
#define surf_extended false
#define surf_upright false

// orb_matcher should be BFMatcher or lshFlannBasedMatcher or desConvertFlannBasedMatcher
#define orb_matcher desConvertFlannBasedMatcher
#define orb_nfeatures 100000
#define orb_scaleFactor 1.2f
#define orb_nlevels 8
#define orb_edgeThreshold 31
#define orb_firstLevel 0
#define orb_WTA_K 2
#define orb_scoreType ORB::HARRIS_SCORE
#define orb_patchSize 31
#define orb_fastThreshold 20

#define surf_cuda_hessianThreshold 100
#define surf_cuda_nOctaves 4
#define surf_cuda_nOctaveLayers 2
#define surf_cuda_extended false
#define surf_cuda_keypointsRatio 0.01f
#define surf_cuda_upright false

// orb_cuda_matcher should be should be BFMatcher or desConvertFlannBasedMatcher
#define orb_cuda_matcher desConvertFlannBasedMatcher
#define orb_cuda_nfeatures 100000
#define orb_cuda_scaleFactor 1.2f
#define orb_cuda_nlevels 8
#define orb_cuda_edgeThreshold 31
#define orb_cuda_firstLevel 0
#define orb_cuda_WTA_K 2
#define orb_cuda_scoreType ORB::HARRIS_SCORE
#define orb_cuda_patchSize 31
#define orb_cuda_fastThreshold 20
#define orb_cuda_blurForDescriptor false

// akaze_cuda_matcher should be should be BFMatcher or desConvertFlannBasedMatcher
#define akaze_matcher desConvertFlannBasedMatcher
#define akaze_descriptor_type AKAZE::DESCRIPTOR_MLDB
#define akaze_descriptor_size 0
#define akaze_descriptor_channels 3
#define akaze_threshold 0.001f
#define akaze_nOctaves 4
#define akaze_nOctaveLayers 4
#define akaze_diffusivity KAZE::DIFF_PM_G2

#endif //FEATURESBENCHMARK_FEATURESPARAMS_H

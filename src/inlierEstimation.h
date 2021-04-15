//
// Created by sem-lab on 2021/4/14.
//

#ifndef FEATURESBENCHMARK_INLIERESTIMATION_H
#define FEATURESBENCHMARK_INLIERESTIMATION_H

#include "usedStruct.h"

double inlierEstiWithHomo(matchedFeatures& _matched_features, const Mat& homography, float inlier_threshold = 0.95);
inliersInfo inlierEstiWithoutHomo(matchedFeatures& _matched_features, float inlier_threshold = 0.95);

#endif //FEATURESBENCHMARK_INLIERESTIMATION_H

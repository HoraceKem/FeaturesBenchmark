//
// Created by sem-lab on 2021/4/13.
//

#ifndef FEATURESBENCHMARK_JSONIO_H
#define FEATURESBENCHMARK_JSONIO_H
#include "json/json.h"
#include "usedStruct.h"

void writeRunningResult(String& save_path, matchedFeatures& featureResult, double timeTaken, inliersInfo& inliers_info);
void writeFilteredMatches(String& save_path, inliersInfo& inliers_info);

#endif //FEATURESBENCHMARK_JSONIO_H

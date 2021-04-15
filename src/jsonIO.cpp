// @Time : 2021/4/13 下午7:37
// @Author : Horace.Kem
// @File: jsonIO.cpp
// @Software: CLion
#include "json/json.h"
#include "usedStruct.h"
#include <fstream>
#include <iostream>

template <typename Iterable>
Json::Value iterable2json(Iterable const& cont) {
    Json::Value v;
    for (auto&& element: cont) {
        v.append(element);
    }
    return v;
}

std::vector<double> homography_mat_to_vector(const Mat& src){
    std::vector<double> dst;
    for(int i=0;i<3;i++){
        for(int j=0; j<3; j++){
            double value = src.at<double>(i,j);
            dst.push_back(value);
        }
    }
    return dst;
}

std::vector<double> point2f_to_vector(const Point2f& pt){
    std::vector<double> dst;
    dst.push_back(pt.x);
    dst.push_back(pt.y);
    return dst;
}

void writeRunningResult(String& save_path, matchedFeatures& featureResult, double timeTaken, inliersInfo& inliers_info) {
    Json::Value root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    root["img1_features"] = featureResult.extractedFeatures1.keyPoints.size();
    root["img2_features"] = featureResult.extractedFeatures2.keyPoints.size();
    root["matched_features"] = featureResult.matches.pair_matches.size();
    root["inliers"] = inliers_info.inliers;
    root["inliers_ratio"] = inliers_info.inliers_ratio;
    root["homography"] = iterable2json(homography_mat_to_vector(inliers_info.homography));
    root["running_time"] = timeTaken;

    std::ofstream ofs;
    ofs.open(save_path);
    assert(ofs.is_open());
    writer->write(root, &ofs);
}

void writeFilteredMatches(String& save_path, inliersInfo& inliers_info) {
    Json::Value root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    Json::Value correspondencePointPairs;
    for(auto & i : inliers_info.filtered_good_matches) {
        Json::Value p1, p2;
        Json::Value pair;
        p1["l"] = iterable2json(point2f_to_vector(i.at(0)));
        p2["l"] = iterable2json(point2f_to_vector(i.at(1)));
        pair["p1"] = p1;
        pair["p2"] = p2;
        correspondencePointPairs.append(pair);
    }
    root["correspondencePointPairs"] = correspondencePointPairs;
    std::ofstream ofs;
    ofs.open(save_path);
    assert(ofs.is_open());
    writer->write(root, &ofs);
}
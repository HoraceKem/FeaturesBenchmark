// @Time : 2021/4/14 下午1:54
// @Author : Horace.Kem
// @File: inlierEstimation.cpp
// @Software: CLion
#include "usedStruct.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

double inlierEstiWithHomo(matchedFeatures& _matched_features, const Mat& homography, float inlier_threshold = 0.95){
    std::vector<DMatch> filtered_good_matches;
    std::vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < _matched_features.matches.matches1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = _matched_features.matches.matches1[i].pt.x;
        col.at<double>(1) = _matched_features.matches.matches1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - _matched_features.matches.matches2[i].pt.x, 2) +
                            pow(col.at<double>(1) - _matched_features.matches.matches2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(_matched_features.matches.matches1[i]);
            inliers2.push_back(_matched_features.matches.matches1[i]);
            filtered_good_matches.emplace_back(new_i, new_i, 0);
        }
    }
    double inlier_ratio = inliers1.size() / (double) _matched_features.matches.matches1.size();
    return inlier_ratio;
}

inliersInfo inlierEstiWithoutHomo(matchedFeatures& _matched_features, float inlier_threshold = 0.95){
    std::vector<Point2f> pts1, pts2;
    for (auto & i : _matched_features.matches.matches1) {
        pts1.push_back(i.pt);
    }
    for (auto & j : _matched_features.matches.matches2) {
        pts2.push_back(j.pt);
    }
    Mat homography = findHomography(pts1, pts2, RANSAC);
    std::vector<std::vector<Point2f>> filtered_good_matches;
    std::vector<KeyPoint> inliers1, inliers2;
    for(size_t i = 0; i < _matched_features.matches.matches1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = _matched_features.matches.matches1[i].pt.x;
        col.at<double>(1) = _matched_features.matches.matches1[i].pt.y;
        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - _matched_features.matches.matches2[i].pt.x, 2) +
                            pow(col.at<double>(1) - _matched_features.matches.matches2[i].pt.y, 2));
        if(dist < inlier_threshold) {
            inliers1.push_back(_matched_features.matches.matches1[i]);
            inliers2.push_back(_matched_features.matches.matches2[i]);
            std::vector<Point2f> pts_pair{_matched_features.matches.matches1[i].pt, _matched_features.matches.matches2[i].pt};
            filtered_good_matches.emplace_back(pts_pair);
        }
    }
    inliersInfo inliers_info;
    inliers_info.inliers = inliers1.size();
    inliers_info.inliers_ratio = inliers1.size() / (double) _matched_features.matches.matches1.size();
    inliers_info.homography = homography;
    inliers_info.filtered_good_matches = filtered_good_matches;
    return inliers_info;
}
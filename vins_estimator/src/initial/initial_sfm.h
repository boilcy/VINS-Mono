#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

#include "../factor/visual_factor.h"

class GlobalSFM
{
public:
    GlobalSFM();
    bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                   const Matrix3d relative_R, const Vector3d relative_T,
                   vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
    bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                          Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                              int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                              vector<SFMFeature> &sfm_f);

    int feature_num;
};
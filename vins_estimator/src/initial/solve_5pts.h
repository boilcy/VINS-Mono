#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>

#include "../feature_manager.h"

class MotionEstimator
{
public:
  static bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
  static bool relativePose(const FeatureManager &f_manager, Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
};
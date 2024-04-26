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

#include "../factor/visual_factor.h"
#include "initial_base.h"

class GlobalLiGT
{
public:
    GlobalLiGT();
    bool construct(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager);

private:
    void build_LTL(const Eigen::Matrix3d *q, std::vector<SFMFeature> &sfm_f, Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr);
    bool solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors);
    void identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors);
    void select_base_views(const std::vector<std::pair<int, Eigen::Vector2d>> &track, const Eigen::Matrix3d *q, int &lbase_view_id, int &rbase_view_id);
    inline Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d &x)
    {
        Eigen::Matrix3d X;
        X << 0, -x(2), x(1),
            x(2), 0, -x(0),
            -x(1), x(0), 0;
        return X;
    }

    int frame_num;
};
#pragma once
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>
#include <memory>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../utility/geometry.hpp"
#include "../utility/opengv_method.hpp"
#include "../feature_manager.h"
#include "initial_base.h"
#include "initial_ligt.h"

/// @brief
class DrtLooselyInit : public LooselyInit
{
public:
    DrtLooselyInit() {}

    Initializer::Status initialize(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager, const std_msgs::Header *Headers, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x) override;
    bool VisualConstruct(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager);
    bool solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs);
};
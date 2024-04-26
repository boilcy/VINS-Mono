#pragma once
#include <iostream>
#include "../utility/utility.h"
#include "../feature_manager.h"
#include "initial_base.h"

class VinsInit : public LooselyInit
{
public:
    enum class VisualConstructStatus
    {
        SUCCESS,
        FEATURE_INSUFFCIENT,
        SFM_FALIURE,
        PNP_INSUFFCIENT,
        PNP_FAILURE,
        UNKNOWN_FAILURE
    };
    VinsInit() {}
    Initializer::Status initialize(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager, const std_msgs::Header *Headers, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &) override;
    bool solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs);
    VisualConstructStatus VisualConstruct(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager, const std_msgs::Header *Headers);
};
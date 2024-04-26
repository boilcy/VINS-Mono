#pragma once
#include <std_msgs/Header.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <map>

#include "../feature_manager.h"
#include "../factor/imu_factor.h"

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    // map<feature_id, xyz_uv_velocity
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    IntegrationBase *pre_integration;
    bool is_key_frame;
};

class Initializer
{
public:
    enum class Status
    {
        SUCCESS,
        ACC_SOLVE_FAILURE,
        GYR_SOLVE_FAILURE,
        SFM_CONSTRUCT_FAILURE,
        LIGT_CONSTRUCT_FAILURE,
        GRAVITY_REFINE_FAILURE,
        GRAVITY_REFINE_MISTAKE,
        UNKNOWN_ERROR,
        DEFAULT
    };
    virtual Status initialize(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager, const std_msgs::Header *Headers, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x) = 0;
};

class LooselyInit : public Initializer
{
public:
    bool RefineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);

    bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x);

    MatrixXd TangentBasis(Vector3d &g0)
    {

        Vector3d b, c;
        Vector3d a = g0.normalized();
        Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();
        c = a.cross(b);
        MatrixXd bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }
};
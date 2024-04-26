#include "initial_drt.h"

struct BiasSolverCostFunctor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasSolverCostFunctor(const std::vector<Eigen::Vector3d> &bearings1,
                          const std::vector<Eigen::Vector3d> &bearings2,
                          const Eigen::Quaterniond &qic,
                          const IntegrationBase *integrate) : _qic(qic)
    {

        jacobina_q_bg = integrate->jacobian.block<3, 3>(O_R, O_BG);
        qjk_imu = integrate->delta_q;
        Eigen::Quaterniond qcjk = _qic.inverse() * qjk_imu;

        // create F things
        for (int i = 0; i < bearings1.size(); i++)
        {
            Eigen::Vector3d f1 = bearings1[i].normalized();

            // Rij.inverse() * fi --- equation 5
            f1 = qcjk.inverse() * f1; // fj'

            Eigen::Vector3d f2 = bearings2[i].normalized();

            f2 = _qic * f2; // fk'

            Eigen::Matrix3d F = f2 * f2.transpose();

            double weight = 1.0;
            xxF_ = xxF_ + weight * f1[0] * f1[0] * F;
            yyF_ = yyF_ + weight * f1[1] * f1[1] * F;
            zzF_ = zzF_ + weight * f1[2] * f1[2] * F;
            xyF_ = xyF_ + weight * f1[0] * f1[1] * F;
            yzF_ = yzF_ + weight * f1[1] * f1[2] * F;
            // zxF_ = zxF_ + weight * f1[2] * f1[0] * F; //opengv
            xzF_ = xzF_ + weight * f1[0] * f1[2] * F; // ROBA
        }
    }

    template <typename T>
    bool operator()(const T *const parameter, T *residual) const
    {

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> deltaBg(parameter);

        Eigen::Matrix<T, 3, 1> jacobian_bg = jacobina_q_bg.cast<T>() * deltaBg;

        Eigen::Matrix<T, 4, 1> qij_tmp;

        ceres::AngleAxisToQuaternion<T>(jacobian_bg.data(), qij_tmp.data());

        Eigen::Quaternion<T> qij(qij_tmp(0), qij_tmp(1), qij_tmp(2), qij_tmp(3));

        Eigen::Matrix<T, 3, 1> cayley = Quaternion2Cayley<T>(qij);

        Eigen::Matrix<T, 1, 3> jacobian;

        T EV = opengv::GetSmallestEVwithJacobian(
            xxF_, yyF_, zzF_, xyF_, yzF_, xzF_, cayley, jacobian);
        residual[0] = EV;

        return true;
    }

    static ceres::CostFunction *
    Create(const std::vector<Eigen::Vector3d> &bearings1,
           const std::vector<Eigen::Vector3d> &bearings2,
           const Eigen::Quaterniond &qic,
           const IntegrationBase *integratePtr)
    {
        return (new ceres::AutoDiffCostFunction<BiasSolverCostFunctor, 1, 3>(
            new BiasSolverCostFunctor(bearings1, bearings2, qic, integratePtr)));
    }

private:
    Eigen::Matrix3d xxF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yyF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d zzF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d xyF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yzF_ = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d xzF_ = Eigen::Matrix3d::Zero();

    Eigen::Matrix3d jacobina_q_bg;
    Eigen::Quaterniond qjk_imu;
    Eigen::Quaterniond _qic;
};

Initializer::Status DrtLooselyInit::initialize(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager, const std_msgs::Header *Headers, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    if (!solveGyroscopeBias(all_image_frame, Bgs))
    {
        return Status::GYR_SOLVE_FAILURE;
    }

    // solve LiGT
    if (!VisualConstruct(all_image_frame, f_manager))
    {
        return Status::LIGT_CONSTRUCT_FAILURE;
    }

    //
    if (!LinearAlignment(all_image_frame, g, x))
    {
        return Status::GRAVITY_REFINE_FAILURE;
    }

    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if (fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return Status::GRAVITY_REFINE_MISTAKE;
    }

    if (!RefineGravity(all_image_frame, g, x))
    {
        return Status::GRAVITY_REFINE_FAILURE;
    }

    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if (s < 0.0)
        return Status::GRAVITY_REFINE_MISTAKE;
    else
        return Status::SUCCESS;
}

bool DrtLooselyInit::solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs)
{
    Vector3d biasg;

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1e-5);

    // image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    // 取相邻keyframe. See DrtVioInit line 360.
    // Num of camera是1
    for (auto it = all_image_frame.begin(); std::next(it) != all_image_frame.end(); ++it)
    {
        ImageFrame current_frame = it->second;

        auto next_it = std::next(it);
        ImageFrame next_frame = next_it->second;

        std::vector<Eigen::Vector3d> fis;
        std::vector<Eigen::Vector3d> fjs;

        std::vector<Eigen::Vector2d> fis_img;
        std::vector<Eigen::Vector2d> fjs_img;

        for (const auto &point1 : current_frame.points)
        {
            if (next_frame.points.find(point1.first) != next_frame.points.end())
            {
                int feature_id = point1.first;
                // Key exists in both ImageFrames
                const auto point2 = next_frame.points[feature_id][0];

                Eigen::Vector3d position1 = point1.second[0].second.block<3, 1>(0, 0);
                Eigen::Vector2d uv1 = point1.second[0].second.block<2, 1>(3, 0);

                Eigen::Vector3d position2 = point2.second.block<3, 1>(0, 0);
                Eigen::Vector2d uv2 = point2.second.block<2, 1>(3, 0);

                fis.push_back(position1);
                fjs.push_back(position2);

                fis_img.push_back(uv1);
                fjs_img.push_back(uv2);
            }
        }

        IntegrationBase *imu = next_frame.pre_integration;
        ceres::CostFunction *eigensolver_cost_function = BiasSolverCostFunctor::Create(fis, fjs, Eigen::Quaterniond(RIC[0]), imu);
        problem.AddResidualBlock(eigensolver_cost_function, loss_function, biasg.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.min_linear_solver_iterations = 10;
    options.gradient_tolerance = 1e-20;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    // options.jacobi_scaling = false;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    try
    {
        ceres::Solve(options, &problem, &summary);
    }
    catch (...)
    {
        ROS_WARN_STREAM("gyroscope bias ceres error " << biasg.transpose());
        return false;
    }

    if (summary.termination_type != ceres::TerminationType::CONVERGENCE)
    {
        ROS_WARN_STREAM("gyroscope bias not converge " << biasg.transpose());
        return false;
    }
    ROS_WARN_STREAM("gyroscope bias initial calibration " << biasg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] = biasg;

    all_image_frame.begin()->second.R = Eigen::Matrix3d::Identity();

    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        auto frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
        Eigen::Matrix3d dRcicj = RIC[0].transpose() * frame_j->second.pre_integration->delta_q.toRotationMatrix() * RIC[0];
        frame_j->second.R = frame_i->second.R * dRcicj;
    }

    return true;
}

bool DrtLooselyInit::VisualConstruct(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager)
{
    GlobalLiGT ligt;
    return ligt.construct(all_image_frame, f_manager);
}
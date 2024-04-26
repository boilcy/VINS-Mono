#include "initial_ligt.h"
#include "../utility/tic_toc.h"

GlobalLiGT::GlobalLiGT() {}

// For DRT, rotation is initialized by gyroscope
bool GlobalLiGT::construct(std::map<double, ImageFrame> &all_image_frame, const FeatureManager &f_manager)
{
    frame_num = all_image_frame.size();
    TicToc t_ligt;

    vector<SFMFeature> sfm_f;
    int num_pts = 0;

    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;

        if (it_per_id.feature_per_frame.size() < 3)
        {
            continue;
        }
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
        ++num_pts;
    }

    Matrix3d Q[frame_num];
    map<double, ImageFrame>::iterator frame_it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++, i++)
    {
        Q[i] = frame_it->second.R;
    }

    Eigen::MatrixXd LTL = Eigen::MatrixXd::Zero(frame_num * 3 - 3, frame_num * 3 - 3);
    Eigen::MatrixXd A_lr = Eigen::MatrixXd::Zero(num_pts, 3 * frame_num);
    build_LTL(Q, sfm_f, LTL, A_lr);

    Eigen::VectorXd evectors = Eigen::VectorXd::Zero(3 * frame_num);

    if (!solve_LTL(LTL, evectors))
    {
        return false;
    }

    identify_sign(A_lr, evectors);

    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++, i++)
    {
        frame_it->second.T = evectors.middleRows<3>(3 * i);
    }
    double time_ligt = t_ligt.toc();

    return true;
}

// sfm_f (feature_id: (frame_id: xy(z=1)))
void GlobalLiGT::build_LTL(const Eigen::Matrix3d *q, std::vector<SFMFeature> &sfm_f, Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr)
{
    int num_view_ = frame_num;
    int track_id = 0;
    for (const auto &feature_per_id : sfm_f)
    {
        const auto &obs = feature_per_id.observation;
        if (obs.size() < 3)
            continue;

        int frame_l = 0;
        int frame_r = 0;

        select_base_views(obs, q, frame_l, frame_r);

        Eigen::MatrixXd tmp_LiGT_vec = Eigen::MatrixXd::Zero(3, num_view_ * 3);

        Eigen::Vector3d pt_l = Eigen::Vector3d(obs[frame_l].second[0], obs[frame_l].second[1], 1);
        Eigen::Vector3d pt_r = Eigen::Vector3d(obs[frame_r].second[0], obs[frame_r].second[1], 1);

        for (const auto &feature_per_frame : obs)
        {
            int frame_i = feature_per_frame.first;
            if (frame_i != frame_l)
            {
                Eigen::Vector3d pt_i = Eigen::Vector3d(feature_per_frame.second[0], feature_per_frame.second[1], 1);
                Eigen::Matrix3d xi_cross = cross_product_matrix(pt_i);
                Eigen::Matrix3d R_cicl = q[frame_i].transpose() * q[frame_l];
                Eigen::Matrix3d R_crcl = q[frame_r].transpose() * q[frame_l];
                Eigen::Vector3d a_lr_tmp_t = cross_product_matrix(R_crcl * pt_l) * pt_r;
                Eigen::RowVector3d a_lr_t = a_lr_tmp_t.transpose() * cross_product_matrix(pt_r);
                A_lr.row(track_id).block<1, 3>(0, frame_l * 3) = a_lr_t * q[frame_r].transpose();
                A_lr.row(track_id).block<1, 3>(0, frame_r * 3) = -a_lr_t * q[frame_r].transpose();
                Eigen::Vector3d theta_lr_vector = cross_product_matrix(pt_r) * R_crcl * pt_l;

                double theta_lr = theta_lr_vector.squaredNorm();

                Eigen::Matrix3d Coefficient_B = xi_cross * R_cicl * pt_l * a_lr_t * q[frame_r].transpose();

                Eigen::Matrix3d Coefficient_C = theta_lr * cross_product_matrix(pt_i) * q[frame_i].transpose();

                Eigen::Matrix3d Coefficient_D = -(Coefficient_B + Coefficient_C);
                tmp_LiGT_vec.setZero();
                tmp_LiGT_vec.block<3, 3>(0, frame_r * 3) += Coefficient_B;
                tmp_LiGT_vec.block<3, 3>(0, frame_i * 3) += Coefficient_C;
                tmp_LiGT_vec.block<3, 3>(0, frame_l * 3) += Coefficient_D;

                Eigen::MatrixXd LTL_l_row = Coefficient_D.transpose() * tmp_LiGT_vec;
                Eigen::MatrixXd LTL_r_row = Coefficient_B.transpose() * tmp_LiGT_vec;
                Eigen::MatrixXd LTL_i_row = Coefficient_C.transpose() * tmp_LiGT_vec;

                {
                    if (frame_l > 0)
                        LTL.middleRows<3>(frame_l * 3 - 3) += LTL_l_row.rightCols(LTL_l_row.cols() - 3);

                    if (frame_r > 0)
                        LTL.middleRows<3>(frame_r * 3 - 3) += LTL_r_row.rightCols(LTL_r_row.cols() - 3);

                    if (frame_i > 0)
                        LTL.middleRows<3>(frame_i * 3 - 3) += LTL_i_row.rightCols(LTL_i_row.cols() - 3);
                }
            }
        }

        ++track_id;
    }
}

bool GlobalLiGT::solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(LTL, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    evectors.bottomRows(V.rows()) = V.col(V.cols() - 1);
    return true;
}

void GlobalLiGT::identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors)
{
    const Eigen::VectorXd judgeValue = A_lr * evectors;
    const int positive_count = (judgeValue.array() > 0.0).cast<int>().sum();
    const int negative_count = judgeValue.rows() - positive_count;
    if (positive_count < negative_count)
    {
        evectors = -evectors;
    }
}

void GlobalLiGT::select_base_views(const std::vector<std::pair<int, Eigen::Vector2d>> &track, const Eigen::Matrix3d *q, int &lbase_view_id, int &rbase_view_id)
{
    double best_criterion_value = -1.;
    std::vector<int> track_id;
    // track_id.reserve(track.size());

    for (const auto &frame : track)
    {
        track_id.push_back(frame.first);
    }

    size_t track_size = track_id.size(); // num_pts_

    // [Step.2 in Pose-only Algorithm]: select the left/right-base views
    for (int i = 0; i < track_size - 1; ++i)
    {
        for (int j = i + 1; j < track_size; ++j)
        {
            const Eigen::Vector3d &i_coord = Eigen::Vector3d(track[i].second[0], track[i].second[1], 1);
            const Eigen::Vector3d &j_coord = Eigen::Vector3d(track[j].second[0], track[j].second[1], 1);

            // R_i is world to camera i
            const Eigen::Matrix3d &R_i = q[i];
            const Eigen::Matrix3d &R_j = q[j];
            // camera i to camera j
            // Rcjw *  Rwci
            const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
            const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

            double criterion_value = theta_ij.norm();

            if (criterion_value > best_criterion_value)
            {

                best_criterion_value = criterion_value;

                if (i < j)
                {
                    lbase_view_id = i;
                    rbase_view_id = i;
                }
                else
                {
                    lbase_view_id = j;
                    rbase_view_id = i;
                }
            }
        }
    }
}
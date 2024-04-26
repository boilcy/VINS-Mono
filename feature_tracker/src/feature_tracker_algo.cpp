#include "feature_tracker_algo.h"

void OriginalFeatureTracking::init()
{
}

void OriginalFeatureTracking::checkEncoding(const cv::Mat &src, cv::Mat &dst)
{
    int channels = src.channels();
    if (channels == 1)
    {
        dst = src.clone();
    }
    else
    {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    }
}

void OriginalFeatureTracking::preProcess(const cv::Mat &src)
{
}

void OriginalFeatureTracking::calcOpticalFlowPyr(const cv::Mat &prevImg, const cv::Mat &nextImg,
                                                 const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                                                 cv::OutputArray status, cv::OutputArray err,
                                                 cv::Size winSize = cv::Size(21, 21), int maxLevel = 3)
{
    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel);
}
void OriginalFeatureTracking::featuresToTrack(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                                              int maxCorners, double qualityLevel,
                                              double minDistance, cv::InputArray mask = cv::noArray())
{
    cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, mask);
}

void LETNetFeatureTracking::init()
{
    score = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_32FC1);
    desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    last_desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    net.load_param(LETNET_PARAM.c_str());
    net.load_model(LETNET_MODEL.c_str());
}

void LETNetFeatureTracking::checkEncoding(const cv::Mat &src, cv::Mat &dst)
{
    int channels = src.channels();
    if (channels == 1)
    {
        ROS_ERROR("LET-NET needs colorful image as input");
    }
    else
    {
        dst = src.clone();
    }
}

void LETNetFeatureTracking::preProcess(const cv::Mat &src)
{
    last_desc = desc.clone();
    cv::Mat img;
    cv::resize(src, img, cv::Size(LET_WIDTH, LET_HEIGHT));
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    // opencv to ncnn
    in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    in.substract_mean_normalize(mean_vals, norm_vals);
    // extract
    ex.input("input", in);
    ex.extract("score", out1);
    ex.extract("descriptor", out2);
    // ncnn to opencv
    out1.substract_mean_normalize(mean_vals_inv, norm_vals_inv);
    out2.substract_mean_normalize(mean_vals_inv, norm_vals_inv);

    //    out1.to_pixels(score.data, ncnn::Mat::PIXEL_GRAY);
    memcpy((uchar *)score.data, out1.data, LET_HEIGHT * LET_WIDTH * sizeof(float));
    cv::Mat desc_tmp(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    out2.to_pixels(desc_tmp.data, ncnn::Mat::PIXEL_BGR);
    desc = desc_tmp.clone();

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
}

void LETNetFeatureTracking::calcOpticalFlowPyr(const cv::Mat &prevImg, const cv::Mat &nextImg,
                                               const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                                               cv::OutputArray status, cv::OutputArray err,
                                               cv::Size winSize = cv::Size(21, 21), int maxLevel = 3)
{
    // resize
    std::vector<cv::Point2f> corners1, corners2;
    int w0 = nextImg.cols;
    int h0 = nextImg.rows;
    float k_w = float(w0) / float(LET_WIDTH);
    float k_h = float(h0) / float(LET_HEIGHT);

    corners1.resize(prevPts.size());
    for (int i = 0; i < int(prevPts.size()); i++)
    {
        corners1[i].x = prevPts[i].x / k_w;
        corners1[i].y = prevPts[i].y / k_h;
    }
    cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2,
                             status, err, cv::Size(21, 21), 5);
    // resize corners2 to forw_pts
    nextPts.resize(corners2.size());
    for (int i = 0; i < int(corners2.size()); i++)
    {
        nextPts[i].x = corners2[i].x * k_w;
        nextPts[i].y = corners2[i].y * k_h;
    }
    // subpixel refinement
    cv::cornerSubPix(gray,
                     nextPts,
                     cv::Size(3, 3),
                     cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                      5, 0.01));
}
void LETNetFeatureTracking::featuresToTrack(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                                            int maxCorners, double qualityLevel,
                                            double minDistance, cv::InputArray mask = cv::noArray())
{
    cv::Mat _mask;
    cv::resize(mask, _mask, cv::Size(LET_WIDTH, LET_HEIGHT));
    f2t(score, corners, maxCorners, 0.0001, (double)MIN_DIST, _mask, 3);
    int w0 = image.cols;
    int h0 = image.rows;
    float k_w = float(w0) / float(LET_WIDTH);
    float k_h = float(h0) / float(LET_HEIGHT);
    for (auto &n_pt : corners)
    {
        n_pt.x *= k_w;
        n_pt.y *= k_h;
    }
    // subpixel refinement
    cv::cornerSubPix(gray,
                     corners,
                     cv::Size(3, 3),
                     cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 5, 0.01));
}

void LETNetFeatureTracking::f2t(cv::InputArray image,
                                cv::OutputArray _corners,
                                int maxCorners,
                                double qualityLevel,
                                double minDistance,
                                const cv::Mat &mask,
                                int blockSize = 3)
{
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && image.sameSize(mask)));

    cv::Mat eig = image.getMat(), tmp;
    double maxVal = 0;
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, mask);
    cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO);
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size imgsize = eig.size();
    std::vector<const float *> tmpCorners;

    cv::Mat Mask = mask;
    for (int y = 1; y < imgsize.height - 1; y++)
    {
        const float *eig_data = (const float *)eig.ptr(y);
        const float *tmp_data = (const float *)tmp.ptr(y);
        const uchar *mask_data = mask.data ? mask.ptr(y) : 0;

        for (int x = 1; x < imgsize.width - 1; x++)
        {
            float val = eig_data[x];
            if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]))
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<cv::Point2f> corners;
    std::vector<float> cornersQuality;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    std::sort(tmpCorners.begin(), tmpCorners.end(), [](const float *a, const float *b)
              { return (*a > *b) ? true : (*a < *b) ? false
                                                    : (a > b); });

    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = eig.cols;
        int h = eig.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f>> grid(grid_width * grid_height);

        minDistance *= minDistance;

        for (i = 0; i < total; i++)
        {
            int ofs = (int)((const uchar *)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width - 1, x2);
            y2 = std::min(grid_height - 1, y2);

            for (int yy = y1; yy <= y2; yy++)
            {
                for (int xx = x1; xx <= x2; xx++)
                {
                    std::vector<cv::Point2f> &m = grid[yy * grid_width + xx];

                    if (m.size())
                    {
                        for (j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if (dx * dx + dy * dy < minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

        break_out:

            if (good)
            {
                grid[y_cell * grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                cornersQuality.push_back(*tmpCorners[i]);

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if (maxCorners > 0 && (int)ncorners == maxCorners)
                    break;
            }
        }
    }
    else
    {
        for (i = 0; i < total; i++)
        {
            cornersQuality.push_back(*tmpCorners[i]);

            int ofs = (int)((const uchar *)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y * eig.step) / sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;

            if (maxCorners > 0 && (int)ncorners == maxCorners)
                break;
        }
    }

    cv::Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}
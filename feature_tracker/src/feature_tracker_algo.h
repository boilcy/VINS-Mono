#pragma once
#include <vector>
#include <memory>
#include <cstdio>
#include <opencv2/opencv.hpp>

#include "parameters.h"
#include "tic_toc.h"
#include "net.h"

class FeatureTrackingAlgorithm
{
public:
    virtual void init() = 0;
    virtual void checkEncoding(const cv::Mat &src, cv::Mat &dst) = 0;
    virtual void preProcess(const cv::Mat &src) = 0;
    virtual void calcOpticalFlowPyr(const cv::Mat &prevImg, const cv::Mat &nextImg,
                                    const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                                    cv::OutputArray status, cv::OutputArray err,
                                    cv::Size winSize, int maxLevel) = 0;
    virtual void featuresToTrack(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                                 int maxCorners, double qualityLevel,
                                 double minDistance, cv::InputArray mask) = 0;
    virtual ~FeatureTrackingAlgorithm() {}
};

class OriginalFeatureTracking : public FeatureTrackingAlgorithm
{
public:
    OriginalFeatureTracking() {}

    void init() override;
    void checkEncoding(const cv::Mat &src, cv::Mat &dst) override;
    void preProcess(const cv::Mat &src) override;
    void calcOpticalFlowPyr(const cv::Mat &prevImg, const cv::Mat &nextImg,
                            const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                            cv::OutputArray status, cv::OutputArray err,
                            cv::Size winSize, int maxLevel) override;
    void featuresToTrack(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                         int maxCorners, double qualityLevel,
                         double minDistance, cv::InputArray mask) override;
};

#define LET_WIDTH 256  // 512
#define LET_HEIGHT 192 // 384

class LETNetFeatureTracking : public FeatureTrackingAlgorithm
{
private:
    cv::Mat gray;
    cv::Mat score;
    cv::Mat desc;
    cv::Mat last_desc;
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
    const float mean_vals_inv[3] = {0, 0, 0};
    const float norm_vals_inv[3] = {255.f, 255.f, 255.f};

    ncnn::Net net;
    ncnn::Mat in;
    ncnn::Mat out1, out2;

public:
    LETNetFeatureTracking() {}

    void init() override;
    void checkEncoding(const cv::Mat &src, cv::Mat &dst) override;
    void preProcess(const cv::Mat &src) override;
    void calcOpticalFlowPyr(const cv::Mat &prevImg, const cv::Mat &nextImg,
                            const std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts,
                            cv::OutputArray status, cv::OutputArray err,
                            cv::Size winSize, int maxLevel) override;
    void featuresToTrack(const cv::Mat &image, std::vector<cv::Point2f> &corners,
                         int maxCorners, double qualityLevel,
                         double minDistance, cv::InputArray mask) override;
    void f2t(cv::InputArray image,
             cv::OutputArray _corners,
             int maxCorners,
             double qualityLevel,
             double minDistance,
             const cv::Mat &_mask, int blockSize);
};
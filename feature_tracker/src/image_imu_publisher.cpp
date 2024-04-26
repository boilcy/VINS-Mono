#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>

const int ROW = 384; // 请替换为实际的图像行数
const int COL = 512; // 请替换为实际的图像列数

void readImage()
{
    ros::NodeHandle nh("~");
    ros::Publisher pub_raw = nh.advertise<sensor_msgs::Image>("/cam0/image_raw", 100);
    ros::Publisher pub_imu = nh.advertise<sensor_msgs::Imu>("/imu0", 100);
    ros::Rate rate(10); // 10 Hz

    const std::string path = "/volsr/datasets/UMA-VI/lab-module-csc-rev_2019-02-05-17-47-57_InOut/";
    const std::string timestamp_path = path + "cam0/data.csv";
    const std::string imu_path = path + "imu0/data.csv";

    // Read timestamp
    std::ifstream timestamp_file(timestamp_path);
    std::string line;
    std::vector<long double> timestamp;
    std::vector<std::string> img_name;
    std::getline(timestamp_file, line); // skip first line
    while (std::getline(timestamp_file, line))
    {
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ',');
        // string to long double
        timestamp.push_back(std::stod(token));
        std::getline(iss, token, ',');
        img_name.push_back(token);
    }
    long double start_time = timestamp[0];
    for (auto &t : timestamp)
    {
        t -= start_time;
    }

    // Read imu
    std::ifstream imu_file(imu_path);
    std::vector<sensor_msgs::Imu> imu_data;
    std::vector<long double> imu_timestamp;
    std::getline(imu_file, line); // skip first line
    while (std::getline(imu_file, line))
    {
        std::istringstream iss(line);
        std::string token;
        std::getline(iss, token, ',');
        // string to long double
        long double t = std::stod(token);
        std::getline(iss, token, ',');
        double wx = std::stod(token);
        std::getline(iss, token, ',');
        double wy = std::stod(token);
        std::getline(iss, token, ',');
        double wz = std::stod(token);
        std::getline(iss, token, ',');
        double ax = std::stod(token);
        std::getline(iss, token, ',');
        double ay = std::stod(token);
        std::getline(iss, token, ',');
        double az = std::stod(token);
        sensor_msgs::Imu imu_msg;
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.angular_velocity.x = wx;
        imu_msg.angular_velocity.y = wy;
        imu_msg.angular_velocity.z = wz;
        imu_msg.linear_acceleration.x = ax;
        imu_msg.linear_acceleration.y = ay;
        imu_msg.linear_acceleration.z = az;
        imu_data.push_back(imu_msg);
        imu_timestamp.push_back(t);
    }
    for (auto &t : imu_timestamp)
    {
        t -= start_time;
    }

    // Read image and publish
    // Publish first image
    ros::Time start = ros::Time::now();
    sensor_msgs::Image img_msg;
    img_msg.header.stamp = start;
    img_msg.header.frame_id = "world";
    img_msg.height = ROW;
    img_msg.width = COL;
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = COL;
    img_msg.data.resize(ROW * COL * 3);

    cv::Mat img = cv::imread(path + "cam0/data/" + img_name[0]);
    cv::resize(img, img, cv::Size(COL, ROW));
    sensor_msgs::Image::ConstPtr img_ptr = cv_bridge::CvImage(img_msg.header, "bgr8", img).toImageMsg();

    pub_raw.publish(img_ptr);

    int pub_index = 1;
    int imu_index = 0;
    while (ros::ok())
    {
        ros::Time now = ros::Time::now();
        ros::Duration d = now - start;
        // d to ns
        long double d_ns = d.sec * 1e9 + d.nsec;
        if (d_ns < imu_timestamp[imu_index])
        {
            continue;
        }
        else
        {
            ros::Duration imu_time;
            imu_data[imu_index].header.stamp = start + imu_time.fromNSec((int64_t)imu_timestamp[imu_index]);
            pub_imu.publish(imu_data[imu_index]);
            imu_index++;
            if (imu_index == imu_timestamp.size())
            {
                break;
            }
        }
        if (d_ns < timestamp[pub_index])
        {
            continue;
        }
        else
        {
            ros::Duration img_time;
            img_msg.header.stamp = start + img_time.fromNSec((int64_t)timestamp[pub_index]);
            img = cv::imread(path + "cam0/data/" + img_name[pub_index], cv::IMREAD_COLOR);
            cv::resize(img, img, cv::Size(COL, ROW));
            img_ptr = cv_bridge::CvImage(img_msg.header, "bgr8", img).toImageMsg();

            pub_raw.publish(img_ptr);
            pub_index++;
            if (pub_index == timestamp.size())
            {
                break;
            }
            ROS_INFO("pub_index: %d", pub_index);
        }
        rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_imu_publisher");
    readImage();
    return 0;
}

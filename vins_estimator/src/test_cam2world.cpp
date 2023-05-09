#include "estimator/cam2world.hpp"
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <boost/filesystem.hpp>

cam2world* tf_cam2world;
cv::Mat K,D;

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat K,D;
    Eigen::Matrix4d leftcam2world_pose;
    cv::Mat image = getImageFromMsg(img_msg);
    if(tf_cam2world->calcPose(image, K, D, leftcam2world_pose)){
        Eigen::Vector3d t = leftcam2world_pose.block<3, 1>(0, 3);
        std::cout << t(0) << " " << t(1) << " " << t(2) << "\n";
    }
    else{
        std::cout << "no aruco found\n";
    }

}

int main(int argc, char **argv)
{
    std::string package_path = ros::package::getPath("vins");
    std::string config_path = (boost::filesystem::path(package_path).parent_path() / "config" / "realsense_d435i" / "realsense_stereo_imu_config.yaml").string();

    YAML::Node config = YAML::LoadFile(config_path);
    YAML::Node aruco_config = config["aruco_markers"]["arucos"];
    config_path = (boost::filesystem::path(package_path).parent_path() / "config" / "realsense_d435i" / "left.yaml").string();
    YAML::Node camera_config = YAML::LoadFile(config_path);
    
    float fx = camera_config["projection_parameters"]["fx"].as<float>();
    float fy = camera_config["projection_parameters"]["fy"].as<float>();
    float cx = camera_config["projection_parameters"]["cx"].as<float>();
    float cy = camera_config["projection_parameters"]["cy"].as<float>();
    float intrinsics_data[9] = {fx, 0.0, cx, 0.0, fy, cy, 0.0,0.0,1.0};

    float k1 = camera_config["distortion_parameters"]["k1"].as<float>();
    float k2 = camera_config["distortion_parameters"]["k2"].as<float>();
    float p1 = camera_config["distortion_parameters"]["p1"].as<float>();
    float p2 = camera_config["distortion_parameters"]["p2"].as<float>();
    float distortion_data[4] = {k1,k2,p1,p2};

    K = cv::Mat(3,3,CV_32FC1,&intrinsics_data);
    D = cv::Mat(1,4,CV_32FC1, &distortion_data);

    tf_cam2world = new cam2world(aruco_config);

    ros::init(argc, argv, "test_cam2world");
    ros::NodeHandle n("~");
    ros::Subscriber sub_img = n.subscribe(config["image0_topic"].as<std::string>(), 1, img_callback);
    ros::spin();

    return 0;
}
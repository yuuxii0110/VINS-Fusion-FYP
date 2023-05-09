#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

class cam2world
{
    public:
        cam2world(YAML::Node aruco_config);
        bool calcPose(const cv::Mat &img, cv::Mat &K, cv::Mat &D, Eigen::Matrix4d &out);

    private:
        Eigen::Matrix4d T_CamWorld_;
        double marker_size_;
        std::unordered_map<int, std::vector<cv::Point3f>> arucos_3d;
        std::unordered_map<int, std::vector<cv::Point2f>> arucos_2d;
        // cv::aruco::ArucoDetector *detector;
        cv::Ptr<cv::aruco::DetectorParameters> parameters_;
        cv::Ptr<cv::aruco::Dictionary> dictionary_;
};

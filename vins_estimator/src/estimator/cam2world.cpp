#include "cam2world.hpp"

cam2world::cam2world(YAML::Node aruco_config)
{
    for(YAML::const_iterator it=aruco_config.begin();it != aruco_config.end();++it) {
        std::vector<cv::Point3f> points;
        for(int i=0; i<4; i++){
            cv::Point3f point;
            point.x = it->second["c"+std::to_string(i)]["x"].as<double>();
            point.y = it->second["c"+std::to_string(i)]["y"].as<double>();
            point.z = it->second["c"+std::to_string(i)]["z"].as<double>();
            points.push_back(point);
            // std::cout << point.x  << " " << point.y << std::endl;
        }
        arucos_3d[it->second["id"].as<int>()] = points;
    }
    parameters_ = cv::aruco::DetectorParameters::create();
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

bool cam2world::calcPose(const cv::Mat &img, cv::Mat &K, cv::Mat &D, Eigen::Matrix4d &out)
{
    cv::Mat frame = img.clone();
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Mat rvec,r,t;
    Eigen::Matrix3d rotmat;
    Eigen::Vector3d t_vec;
    cv::aruco::detectMarkers(img, dictionary_, markerCorners, markerIds, parameters_, rejectedCandidates);
    arucos_2d.clear();
    for(unsigned int i=0; i<markerIds.size(); i++){
        arucos_2d[markerIds[i]] = markerCorners[i];
    }

    if (markerIds.size() > 0) {
      // Draw the markers on the frame
        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
        for (const auto &aruco : arucos_2d) {
            cv::circle(frame, aruco.second[2], 5, cv::Scalar(200), 2);
        }
    }

    // Display the frame
    cv::imshow("frame", frame);
    cv::waitKey(1);

    std::vector<int> common_keys;
    for(auto el:arucos_3d){
        if(arucos_2d.count(el.first)){
            common_keys.push_back(el.first);
        }
    }

    if(common_keys.size()){
        std::vector<cv::Point2f> pts2d;
        std::vector<cv::Point3f> pts3d;
        for(auto el:common_keys){
            for(unsigned int i=0; i<4;i++){
                pts2d.push_back(arucos_2d[el][i]);
                pts3d.push_back(arucos_3d[el][i]);
            }
        }
        if (cv::solvePnP(pts3d, pts2d, K, D, rvec, t)){
            cv::Rodrigues(rvec, r);
            cv::cv2eigen(r, rotmat);
            t_vec(0) = t.at<double>(0,0);
            t_vec(1) = t.at<double>(0,1);
            t_vec(2) = t.at<double>(0,2);

            out = Eigen::Matrix4d::Identity();
            out.block<3, 3>(0, 0) = rotmat.transpose();
            out.block<3,1>(0, 3) = -1*rotmat.transpose()*t_vec;
            std::cout << "(x,y,z): " << out(0,3) <<","<< out(1,3) <<","<< out(2,3) << "\n"<< std::endl;
            return true;
        }
        else{
            return false;
        }
    }
    return false;
}
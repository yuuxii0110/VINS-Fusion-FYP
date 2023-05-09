#include "estimator/unet_inference.hpp"
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <boost/filesystem.hpp>


std::vector<uint8_t> colors{0,255};

int main(int argc, char **argv){
    std::string package_path = ros::package::getPath("vins");
    std::string config_path = (boost::filesystem::path(package_path) / "model_config" / "models.yaml").string();
    YAML::Node config = YAML::LoadFile(config_path);

    std::string model_path = (boost::filesystem::path(package_path) / "models" / config["model_name"].as<std::string>() ).string();
    std::string imgs_path = (boost::filesystem::path(package_path) / "models" / "test_imgs").string();

    UnetInferece unetModel(config, model_path);

    float rest_time = config["vis_rest_time_ms"].as<float>();
    bool auto_play = config["auto_play"].as<int>();

    int key;

    for (const auto & entry : boost::filesystem::directory_iterator(imgs_path)){
        std::string img_path = entry.path().string();
        cv::Mat test_image = cv::imread(img_path, 0);
        cv::Mat out_image(test_image.rows, test_image.cols, CV_8UC1);
        
        size_t last_slash_pos = img_path.find_last_of("/");
        std::string filename = img_path.substr(last_slash_pos + 1);
        size_t dot_pos = filename.find_last_of(".");
        std::string basename = filename.substr(0, dot_pos);
        boost::filesystem::path file_path(img_path);
        boost::filesystem::path grandparent_path = file_path.parent_path().parent_path();

        std::string maskname = grandparent_path.string() + "/test_masks/" + basename + "_mask.png";
        cv::Mat out = unetModel.getSegmentedClass(test_image);
        
        for(int i=0; i < out.rows; i++)
        {
            for(int j = 0; j < out.cols; j++){
                int pixel_class = out.at<uint8_t>(cv::Point(j,i));
                out_image.at<uint8_t>(cv::Point(j,i)) = colors.at(pixel_class);
            }
        }

        cv::imwrite(maskname, out_image);
        cv::Mat out_mask;
        addWeighted( test_image, 0.5, out_image, 0.5, 0.0, out_mask);
        
        cv::imshow("output",out_mask);
        if(!auto_play){
            cv::waitKey(0);
        }
        else{
            key = cv::waitKey(rest_time);
            if(key == 27){
                key = cv::waitKey(0);
            }
        }

    }

    cv::destroyAllWindows();
    return 0;
}

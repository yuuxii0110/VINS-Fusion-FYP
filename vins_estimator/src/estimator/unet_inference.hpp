#ifndef MODEL_INFERENCE_HPP
#define MODEL_INFERENCE_HPP

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <time.h>
#include <yaml-cpp/yaml.h>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

class UnetInferece{
    public:
        UnetInferece(YAML::Node config, std::string model_path);
        cv::Mat getSegmentedClass(cv::Mat &original_image_mat);

    private:
        std::string model_path_;
        int time_verbose_;
        char* inputName;
        char* outputName;
        bool use_cuda_, edge_assiatance_;
        float canny_up_, canny_down_;
        float rescale_factor_, conf_threshold_;
        size_t inputTensorSize, outputTensorSize, linearlize_vecsize;
        int num_class_, default_class_ = 0;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<int64_t> inputDims, outputDims;
        Ort::Session *session;
        Ort::Env *env;
        clock_t total_begin_time, begin_time2,begin_time3;

        std::vector<cv::Mat> UnetProcessInput(cv::Mat &original_mat);

};

#endif
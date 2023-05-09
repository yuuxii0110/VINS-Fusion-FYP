#ifndef SVM_INFERENCE_HPP
#define SVM_INFERENCE_HPP

#include <onnxruntime_cxx_api.h>
#include "unet_inference.hpp"

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

// template <typename T>
// T vectorProduct(const std::vector<T>& v)
// {
//     return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
// }

class SvmInference{
    public:
        SvmInference(std::string model_path, int normalization_flag);
        int GetImagePatchClass(cv::Mat &original_image_mat);

    private:
        std::string model_path_;
        char* inputName;
        char* outputName;
        int normalization_flag_; //0: no normalization, 1: mean_substraction, 2: z-norm
        size_t inputTensorSize, outputTensorSize;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<int64_t> inputDims, outputDims;
        Ort::Session *session;
        Ort::Env *env;

        std::vector<float> SvmProcessInput(cv::Mat image_patch);

};
#endif
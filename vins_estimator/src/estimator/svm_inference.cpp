#include "svm_inference.hpp"

std::vector<float> SvmInference::SvmProcessInput(const cv::Mat patch)
{
    cv::Mat image_patch = patch.clone();
    image_patch.convertTo(image_patch, CV_32FC1);
    int rows = image_patch.rows, cols = image_patch.cols;
    image_patch = image_patch.reshape(1, rows * cols);
    std::vector<float> linearlize_patch(image_patch.cols * image_patch.rows);
    
    float *ptr = image_patch.ptr<float>();
    for (int i = 0; i < image_patch.cols * image_patch.rows; i++) {
        linearlize_patch[i] = ptr[i];
    }
    std::vector<float> normalized_patch(linearlize_patch);

    if(normalization_flag_){
        float mean = 0, sd = 0;
        for (float val : linearlize_patch) {
            mean += val;
        }
        mean /= linearlize_patch.size();
        for (unsigned long i = 0; i < linearlize_patch.size(); i++) {
            normalized_patch[i] = (linearlize_patch[i] - mean);
        }

        if(normalization_flag_==2){
            for (float val : linearlize_patch) {
                sd += (val - mean) * (val - mean);        
            }
            sd = sqrt(sd / linearlize_patch.size());
            if (sd == 0) {
                sd = 1e-9;
            }
            for (unsigned long i = 0; i < normalized_patch.size(); i++) {
                normalized_patch[i] = (normalized_patch[i])/sd;
            }
        }
    }
    return normalized_patch;
}

SvmInference::SvmInference(std::string model_path, int normalization_flag)
{
    std::cout << "loading svm_...\n";
    normalization_flag_ = normalization_flag;
    model_path_ = model_path;
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    std::string instanceName{"svm-inference"};
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    session = new Ort::Session(*env, model_path_.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();
    inputName = session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    inputDims = inputTensorInfo.GetShape();
    outputName = session->GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    outputDims = outputTensorInfo.GetShape();

    // std::cout << inputDims.at(0) << " " << inputDims.at(1) << " " <<std::endl;
    // std::cout << outputDims.at(0) <<" " << outputDims.at(1) <<" " <<std::endl;
    inputDims.at(0) = 1;
    outputDims.at(0) = 1;

    inputTensorSize = vectorProduct(inputDims);
    outputTensorSize = vectorProduct(outputDims);
    
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input tensor size: " << inputTensorSize << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output tensor size: " << outputTensorSize << std::endl;
}

int SvmInference::GetImagePatchClass(cv::Mat &original_image_mat)
{
    std::vector<float> normalized_patch = SvmProcessInput(original_image_mat);
    std::vector<float> inputTensorValues(inputTensorSize);
    std::copy(normalized_patch.begin(), normalized_patch.end(), inputTensorValues.begin());
    std::vector<int64_t> outputTensorValues(outputTensorSize);
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors; 
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size())); 
        
    outputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), inputTensors.size(), outputNames.data(),
                outputTensors.data(), outputTensors.size());

    auto output_data = outputTensors[0].GetTensorMutableData<float>();
    int prediction = (output_data[0] > 0) ? 1 : 0;
    return prediction;
}
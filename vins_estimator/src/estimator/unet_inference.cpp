#include "unet_inference.hpp"

UnetInferece::UnetInferece(YAML::Node config, std::string model_path){
    std::cout << "loading unet_...\n";
    model_path_ = model_path;
    use_cuda_ = config["use_cuda"].as<int>();
    edge_assiatance_ = config["use_canny"].as<int>();
    if (edge_assiatance_){
        canny_down_ = config["canny_down"].as<float>();
        canny_up_ = config["canny_up"].as<float>();
    }
    rescale_factor_ = config["rescale_factor"].as<float>();
    conf_threshold_ = config["conf_threshold"].as<float>();

    time_verbose_ = config["subtime_verbose"].as<int>() + 2*config["total_time_verbose"].as<int>();
    std::cout << time_verbose_ << std::endl;
    std::string instanceName{"unet-inference"};
    env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (use_cuda_)
    {
        std::cout << "using device cuda!\n";
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Ort::Session session(env, model_path_.c_str(), sessionOptions);
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

    std::cout << inputDims.at(0) << " " << inputDims.at(1) << " " <<inputDims.at(2) << " " << inputDims.at(3) <<std::endl;
    std::cout << outputDims.at(0) <<" " << outputDims.at(1) <<" " << outputDims.at(2) <<" " << outputDims.at(3) <<std::endl;

    //fixing batch size to 1
    inputDims.at(0) = 1;
    outputDims.at(0) = 1;


    inputTensorSize = vectorProduct(inputDims);
    outputTensorSize = vectorProduct(outputDims);
    num_class_ = outputDims.at(1);

    //linearlize_vecsize -> W*H
    linearlize_vecsize = outputTensorSize/num_class_;
    
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input tensor size: " << inputTensorSize << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output tensor size: " << outputTensorSize << std::endl;
    std::cout << "Num of class: " << num_class_ << std::endl;
}

cv::Mat UnetInferece::getSegmentedClass(cv::Mat &original_image_mat){
    if(time_verbose_){
        total_begin_time = clock(); 
    }

    std::vector<cv::Mat> processed_imgs = UnetProcessInput(original_image_mat);

    if(time_verbose_%2){
        std::cout << "preprocessing time " << float( clock () - total_begin_time ) /  CLOCKS_PER_SEC << std::endl;
    }

    std::vector<float> inputTensorValues(inputTensorSize);
    int num_of_channel = processed_imgs.size();
    for(int i=0; i< num_of_channel; i++){
        std::copy(processed_imgs[i].begin<float>(), processed_imgs[i].end<float>(), inputTensorValues.begin() + i*linearlize_vecsize);
    }
        
    std::vector<float> outputTensorValues(outputTensorSize);
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;    
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size())); 
        
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));
    
    if(time_verbose_%2){
        begin_time2 = clock();
    }

    session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), inputTensors.size(), outputNames.data(),
                outputTensors.data(), outputTensors.size());

    if(time_verbose_%2){
        std::cout << "inference time: " << float( clock () - begin_time2 ) /  CLOCKS_PER_SEC << std::endl;
        begin_time3 = clock();
    }

    cv::Mat mask(inputDims.at(2), inputDims.at(3), CV_8UC1); 
    for (size_t i = 0; i < linearlize_vecsize; i++)
    {
        //softmax
        float expSum = 0;
        float out, max_out=0;
        int pred_class=0;

        for(int j=0; j<num_class_;j++){
            out = std::exp(outputTensorValues.at(i+ j*linearlize_vecsize));
            expSum += out;
            if(out > max_out){
                max_out = out;
                pred_class = j;
            }
        }

        int row_id = (int) (i/linearlize_vecsize);
        int column_id = i - row_id*linearlize_vecsize;
        
        mask.at<uint8_t>(cv::Point(column_id,row_id)) = default_class_;
        if(pred_class != default_class_ && max_out/expSum >= conf_threshold_){
            mask.at<uint8_t>(cv::Point(column_id,row_id)) = pred_class;
        }
    }
    int rows = mask.rows /rescale_factor_;
    int cols = mask.cols /rescale_factor_;
    cv::Size s(cols,rows);
    cv::Mat outMask(rows, cols, CV_8UC1);
    cv::resize(mask, outMask, s);

    if(time_verbose_%2){
        std::cout << "post process time: " << float( clock () - begin_time3 ) /  CLOCKS_PER_SEC << std::endl;
    }


    if(time_verbose_ > 1){
        if (use_cuda_){
            std::cout << "inference time with gpu: " << float( clock () - total_begin_time ) /  CLOCKS_PER_SEC << "\n";
        }
        else{
            std::cout << "inference time with cpu: " << float( clock () - total_begin_time ) /  CLOCKS_PER_SEC << "\n";
        }
    }

    return outMask;
}

std::vector<cv::Mat> UnetInferece::UnetProcessInput(cv::Mat &original_mat){
    std::vector<cv::Mat> out;
    cv::Mat resized_image, canny_edges;
    int rows = original_mat.rows *rescale_factor_;
    int cols = original_mat.cols *rescale_factor_;
    cv::Size s(cols,rows);
    cv::resize(original_mat, resized_image, s);

    if(edge_assiatance_){
        cv::Canny(resized_image,canny_edges,canny_down_,canny_up_);
        resized_image.convertTo(resized_image, CV_32F, 1.0 / 255, 0);
        out.push_back(resized_image);
        canny_edges.convertTo(canny_edges, CV_32F, 1.0 / 255, 0);
        out.push_back(canny_edges);
    }
    else{
        resized_image.convertTo(resized_image, CV_32F, 1.0 / 255, 0);
        out.push_back(resized_image);
    }

    //normalization
    return out;
}

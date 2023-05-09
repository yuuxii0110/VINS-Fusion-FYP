#include "ground_feature_fitter.hpp"

GroundPlaneFitter::GroundPlaneFitter(int max_queue_size, int min_fit_size, std::vector<float> weights): 
max_queue_size_(max_queue_size), 
min_fit_size_(min_fit_size),
weights_(weights)
{
    window_size_ = weights_.size();
}

void GroundPlaneFitter::InsertVinsPoint(int feature_id, Eigen::Vector3d pt){
    if((int)features.size() > max_queue_size_){
        features.pop_front();
    }
    for (std::deque<std::pair<int, Eigen::Vector3d>>::iterator it = features.begin(); it != features.end(); ++it) {
        if (it->first == feature_id) {
            features.erase(it);
            break;
        }
    }

    features.push_back(std::make_pair(feature_id, pt));
}

std::vector<float> GroundPlaneFitter::InsertGroundPlane(float a, float b, float c, float d){
    a_params.push_back(a);
    b_params.push_back(b);
    c_params.push_back(c);
    d_params.push_back(d);
    float sum = 0;
    std::vector<float> tmp_storage={0.0,0.0,0.0,0.0};
    for(int i=(int)a_params.size()-1; i>0; i--){
        float weight = weights_[i];
        sum += weight;
        tmp_storage[0] += weight*a_params[i];
        tmp_storage[1] += weight*b_params[i];
        tmp_storage[2] += weight*c_params[i];
        tmp_storage[3] += weight*d_params[i];
    }

    for(int i=0; i<(int)tmp_storage.size(); i++){
        tmp_storage[i] /= sum;
    }
    a = tmp_storage[0];
    b = tmp_storage[1];
    c = tmp_storage[2];
    d = tmp_storage[3];
    double mag = std::sqrt(std::pow(a,2)+ std::pow(b,2)+ std::pow(c,2));
    int last_idx = a_params.size()-1;
    a_params[last_idx] = tmp_storage[0]/mag;
    b_params[last_idx] = tmp_storage[1]/mag;
    c_params[last_idx] = tmp_storage[2]/mag;
    d_params[last_idx] = tmp_storage[3]/mag;
    std::vector<float> out = {a_params[last_idx], b_params[last_idx], c_params[last_idx], d_params[last_idx]};
    if((int)a_params.size()>=window_size_){
        a_params.erase(a_params.begin());
        b_params.erase(b_params.begin());
        c_params.erase(c_params.begin());
        d_params.erase(d_params.begin());
    }
    return out;

}



bool GroundPlaneFitter::fitPlane(Eigen::Vector4d &plane_coef){
    int numData = features.size();
    if(numData < min_fit_size_){
        return false;
    }

    // Copy the original queue into a vector
    std::vector<std::pair<int, Eigen::Vector3d>> queueElements(numData);
    std::copy(features.begin(), features.end(), queueElements.begin());

    Eigen::MatrixXd data(numData, 3);
    for (int i = 0; i < numData; i++) {
        data.row(i) = queueElements[i].second.transpose();
    }

    Eigen::Vector3d mean = data.colwise().mean();

    data.col(0).array() -= mean(0);
    data.col(1).array() -= mean(1);
    data.col(2).array() -= mean(2);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(data, Eigen::ComputeFullV);

    float a = svd.matrixV().col(2)(0);
    float b = svd.matrixV().col(2)(1);
    float c = svd.matrixV().col(2)(2);
    double mag = std::sqrt(std::pow(a,2)+ std::pow(b,2)+ std::pow(c,2));
    a /= mag;
    b /= mag;
    c /= mag;
    if(c < 0){
        a *= -1;
        b *= -1;
        c *= -1;
    }
    float d = -(a * mean(0) + b * mean(1) + c * mean(2));
    InsertGroundPlane(a,b,c,d);
    plane_coef = {a,b,c,d};
    return true;
}
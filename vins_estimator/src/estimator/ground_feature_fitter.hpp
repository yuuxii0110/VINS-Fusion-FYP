#include <eigen3/Eigen/Dense>
#include <iostream>
#include <deque>
#include <vector>

class GroundPlaneFitter{
    public:
        GroundPlaneFitter(int max_queue_size, int min_fit_size, std::vector<float> weights);
        void InsertVinsPoint(int feature_id, Eigen::Vector3d pt);
        std::vector<float> InsertGroundPlane(float a, float b, float c, float d);
        bool fitPlane(Eigen::Vector4d &plane_coef);
        

    private:
        int max_queue_size_ = 100;
        int min_fit_size_ = 50;
        std::vector<float> weights_;
        int window_size_;
        std::deque<std::pair<int, Eigen::Vector3d>> features;
        std::vector<float> a_params, b_params, c_params, d_params;
};





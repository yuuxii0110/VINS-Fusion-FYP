#include "estimator/data_processor.hpp"

int main() {
    std::vector<std::pair<double, double>> vec = { {0.0, 0.0}, {2.0, 1.0}, {4.0, 2.0}, {6.0, 3.0} };
    std::vector<std::pair<double, double>> throttle = { {1.1, 10}, {3.4, 60}, {5.4, 65}, {5.8, 63}, {6.5, 65},{7.2, 71} };
    DataProcessor processor1, processor2;
    processor1.values_ = vec;
    processor2.values_ = throttle;
    std::pair<double, double> acc;
    std::pair<double, double> vel;
    std::pair<double, double> thr;
    double mid_time;
    acc = processor1.ExtractData(DataProcessor::DERIVATIVE);
    vel = processor1.ExtractData(DataProcessor::MIDDLE);
    thr = processor2.ExtractData(DataProcessor::CLOSEST, vel.first);
    std::cout << acc.second << std::endl;
    std::cout << vel.second << std::endl;
    std::cout << thr.second << std::endl;
    
}

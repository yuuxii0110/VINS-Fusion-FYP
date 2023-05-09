#pragma once

#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <iostream>
#include <memory>
#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>

class DataProcessor {
public:
    DataProcessor() {
    }
        
    void InsertData(std::pair<double, double> data) {
        values_.push_back(data);
    }

    std::pair<double, double> ExtractData(const int action, const double time = std::numeric_limits<double>::quiet_NaN()){
        std::pair<double, double> out = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        int n = values_.size();
        if(n){
            if(action == DERIVATIVE){
                // Create Eigen matrices for x and y values
                Eigen::MatrixXd X(n, 2);
                Eigen::VectorXd y(n);
                // Fill matrices with x and y values
                for (int i = 0; i < n; i++) {
                    X(i, 0) = values_[i].first;
                    X(i, 1) = 1.0;
                    y(i) = values_[i].second;
                }
                
                // Calculate least squares solution
                Eigen::VectorXd beta = X.colPivHouseholderQr().solve(y);
                
                // The acceleration is the slope of the line (the first element of beta)
                out.second  = beta(0);
            }
            else if(action == MIDDLE){
                double mid_time = (values_.begin()->first + values_.end()->first)/2;
                return FindClosest(mid_time);
            }
            else if(action == CLOSEST && !std::isnan(time)){
                return FindClosest(time);
            }
            else if(action == AVERAGE){
                double sum = 0.0;
                for (const auto& p : values_) {
                    sum += p.second;
                }
                out.second = sum / values_.size();
            }
        }
        // values_.clear();
        return out;
    }

    const static int DERIVATIVE = 0; //
    const static int MIDDLE = 1; //extract the elements in the middle of [start, end] time
    const static int CLOSEST = 2; //for data which has high variance
    const static int AVERAGE = 3; //for data which has low variance  
    std::vector<std::pair<double, double>> values_;
private:
    
    std::pair<double, double> FindClosest(double xk) {
        auto it = std::lower_bound(values_.begin(), values_.end(), std::make_pair(xk, 0.0),
                                [](const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
                                    return p1.first < p2.first;
                                });
        if (it == values_.end()) {
            return values_.back();
        } else if (it == values_.begin()) {
            return values_.front();
        } else {
            auto it_prev = it - 1;
            if (xk - it_prev->first < it->first - xk) {
                return *it_prev;
            } else {
                return *it;
            }
        }
    }
};

#endif

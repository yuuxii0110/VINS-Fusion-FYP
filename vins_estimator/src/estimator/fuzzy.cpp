#include "fuzzy.hpp"

FuzzyInferenceSystem::FuzzyInferenceSystem(){};

double FuzzyInferenceSystem::getWeightage(double vel, double nums) {
    fuzzifyInputs(vel, nums);
    applyRules();
    return CoGDefuzzify();
}

void FuzzyInferenceSystem::fuzzifyInputs(double vel, double nums) {
    input1_values_ = {
        velocity_slow_.evaluate(vel),
        velocity_normal_.evaluate(vel),
        velocity_fast_.evaluate(vel),
    };
    input2_values_ = {
        num_features_less_.evaluate(nums),
        num_features_normal_.evaluate(nums),
        num_features_many_.evaluate(nums)
    };
}

void FuzzyInferenceSystem::applyRules() {
    output_values_.resize(outputs_.size());
    std::fill(output_values_.begin(), output_values_.end(), 0.0);
    for (auto& rule : rules_) {
        double w = std::min(input1_values_[rule.antecedents[0]], input2_values_[rule.antecedents[1]]);
        output_values_[rule.consequent] = std::max(output_values_[rule.consequent], w * rule.weight);
    }
}

double FuzzyInferenceSystem::CoGDefuzzify(double step_size) {
    double nominator = 0.0;
    double denominator = 0.0;
    int total_iteration = (max_-min_)/step_size + 1;
    double cur_val = min_;
    double weight;
    double height;

    for(int i=0; i<total_iteration; i++){
        if(cur_val< intercept1_){
            height = std::min(output_values_[0], outputs_[0].evaluate(cur_val));
        }

        else if(cur_val < intercept2_){
            height = std::min(output_values_[1], outputs_[1].evaluate(cur_val));
        }

        else{
            height = std::min(output_values_[2], outputs_[2].evaluate(cur_val));
        }
        weight = cur_val*height;
        nominator += weight;
        denominator += height;
        cur_val += step_size;
    }
    return nominator / denominator;   
}

// int main() {
//     FuzzyInferenceSystem fis;
//     double vel, features;
//     std::cin >> vel >> features;
//     double weightage = fis.getWeightage(vel, features);
//     std::cout << "Weightage: " << weightage << std::endl;
//     return 0;
// }
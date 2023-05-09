#include "custom_lossfunc.hpp"

void Weighted_HuberLoss::Evaluate(double s, double rho[3]) const {
    if (s > b_) {
        // Outlier region.
        // 'r' is always positive.
        const double r = sqrt(s);
        rho[0] = weight_* (2 * a_ * r - b_);
        rho[1] = weight_* std::max(std::numeric_limits<double>::min(), a_ / r);
        rho[2] = - weight_* (rho[1] / (2 * s));
    } else {
        // Inlier region.
        rho[0] = weight_*s;
        rho[1] = weight_*1;
        rho[2] = 0;
    }
}
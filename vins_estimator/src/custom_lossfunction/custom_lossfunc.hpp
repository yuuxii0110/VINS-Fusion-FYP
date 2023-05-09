#include <ceres/ceres.h>

//this customized lossfunction is used to emphasize ground features in the optimization
class Weighted_HuberLoss : public ceres::LossFunction {
public:
    explicit Weighted_HuberLoss(double a, double weight) : a_(a), b_(a * a), weight_(weight) { }
    virtual void Evaluate(double, double*) const;

private:
    const double a_;
    // b = a^2.
    const double b_;
    const double weight_;
};
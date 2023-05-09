#include <iostream>
#include <vector>
#include <algorithm>

class MembershipFunction {
public:
    virtual ~MembershipFunction() {}
    virtual double evaluate(double x) const {
        return 0.0;
    }
};

class TriangularMembershipFunction : public MembershipFunction {
public:
    TriangularMembershipFunction(double a, double b, double c) :
        a_(a), b_(b), c_(c) {}

    double evaluate(double x) const override {
        if (x <= a_ || x >= c_) {
            return 0.0;
        } else if (x > a_ && x <= b_) {
            return (x - a_) / (b_ - a_);
        } else if (x > b_ && x < c_) {
            return (c_ - x) / (c_ - b_);
        } else {
            return 0.0;
        }
    }

    double getA(){ return a_;}
    double getB(){ return b_;}
    double getC(){ return c_;}

private:
    double a_, b_, c_;
};

class TrapezoidalMembershipFunction : public MembershipFunction {
public:
    TrapezoidalMembershipFunction(double a, double b, double c, double d) :
        a_(a), b_(b), c_(c), d_(d) {}

    double evaluate(double x) const override {
        if (x <= a_ || x >= d_) {
            return 0.0;
        } else if (x > a_ && x <= b_) {
            return (x - a_) / (b_ - a_);
        } else if (x > b_ && x <= c_) {
            return 1.0;
        } else if (x > c_ && x < d_) {
            return (d_ - x) / (d_ - c_);
        } else {
            return 0.0;
        }
    }

private:
    double a_, b_, c_, d_;
};

class FuzzyInferenceSystem {
public:
    FuzzyInferenceSystem();

    double getWeightage(double vel, double nums);

private:
    double weightage_;
    double max_ = 10.0;
    double min_ = 1.0;
    double intercept1_ = 4;
    double intercept2_ = 7;

    struct FuzzyRule {
        std::vector<int> antecedents;
        int consequent;
        double weight;
    };

    TrapezoidalMembershipFunction velocity_slow_ = TrapezoidalMembershipFunction(-1000, -100, 0.2, 0.8);
    TriangularMembershipFunction velocity_normal_ = TriangularMembershipFunction(0.4, 1, 1.6);
    TrapezoidalMembershipFunction velocity_fast_ = TrapezoidalMembershipFunction(1.2, 1.8, 100, 1000);

    TrapezoidalMembershipFunction num_features_less_ = TrapezoidalMembershipFunction(-1000, -100, 5, 10);
    TriangularMembershipFunction num_features_normal_ = TriangularMembershipFunction(5 ,10, 15);
    TrapezoidalMembershipFunction num_features_many_ = TrapezoidalMembershipFunction(10, 15, 100, 1000);


    TriangularMembershipFunction weightage_low_ = TriangularMembershipFunction(-1000,min_, intercept1_);
    TriangularMembershipFunction weightage_normal_ = TriangularMembershipFunction(intercept1_, (intercept1_+intercept2_)/2, intercept2_);
    TriangularMembershipFunction weightage_high_ = TriangularMembershipFunction(intercept2_, max_, 1000.0);

    std::vector<FuzzyRule> rules_ = {
        {{0, 0}, 1, 1.0},
        {{1, 0}, 1, 1.0},
        {{2, 0}, 0, 1.0},
        {{0, 1}, 2, 1.0},
        {{1, 1}, 1, 1.0},
        {{2, 1}, 0, 1.0},
        {{0, 2}, 1, 1.0},
        {{1, 2}, 1, 1.0},
        {{2, 2}, 0, 1.0}
    };

    std::vector<MembershipFunction> inputs_ = {
        velocity_slow_, velocity_normal_, velocity_fast_, 
        num_features_less_, num_features_normal_, num_features_many_
    };

    std::vector<TriangularMembershipFunction> outputs_ = {
        weightage_low_, weightage_normal_, weightage_high_
    };

    std::vector<double> input1_values_, input2_values_;
    std::vector<double> output_values_;
    
    void fuzzifyInputs(double vel, double nums);
    void applyRules();

    double CoGDefuzzify(double step_size=0.1);
};



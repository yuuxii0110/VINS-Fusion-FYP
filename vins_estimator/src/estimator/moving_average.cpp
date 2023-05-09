#include "moving_average.hpp"

MovingAverage::MovingAverage(int window_size)
  : window_size_(window_size), count_(0), sum_(0.0)
{}

void MovingAverage::add_value(double value) {
  if (values_.size() == window_size_) {
    sum_ -= values_.front();
    values_.pop();
  }
  values_.push(value);
  sum_ += value;
  if(count_ < window_size_)
    count_++;
}

double MovingAverage::get_average() const {
  return sum_ / count_;
}

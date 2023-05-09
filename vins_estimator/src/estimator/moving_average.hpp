#ifndef MOVING_AVERAGE_H
#define MOVING_AVERAGE_H

#include<queue>

class MovingAverage {
public:
  MovingAverage(int window_size=3);
  void add_value(double value);
  double get_average() const;

private:
  int window_size_;
  int count_;
  double sum_;
  std::queue<double> values_;
};

#endif

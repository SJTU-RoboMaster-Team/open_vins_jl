/*
 * Small causal vibration pre-filter for the T265 united IMU stream.
 *
 * The gyro path is a second-order Butterworth low-pass.  The first sample
 * seeds the state so a stationary startup has no zero-to-bias transient.  The
 * accelerometer path is only a short median filter for isolated sample
 * glitches; it is deliberately not a low-pass because the T265 accelerometer
 * is firmware-interpolated from a much lower native rate.
 */
#ifndef OV_CORE_IMU_PREFILTER_H
#define OV_CORE_IMU_PREFILTER_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace ov_core {

class ImuPreFilter {
public:
  void configure(bool gyro_enable, double fs_hz, double cutoff_hz, int acc_window,
                 bool notch_enable = false, double notch_hz = 93.0,
                 double notch_q = 8.0) {
    if (acc_window != 0 && (acc_window < 3 || (acc_window % 2) == 0))
      throw std::invalid_argument("acc_median_window must be zero or odd >= 3");
    if (gyro_enable && !(fs_hz > 0.0 && cutoff_hz > 0.0 && cutoff_hz < fs_hz * 0.5))
      throw std::invalid_argument("invalid gyro low-pass frequency");
    if (notch_enable && !(fs_hz > 0.0 && notch_hz > 0.0 && notch_hz < fs_hz * 0.5 && notch_q > 0.0))
      throw std::invalid_argument("invalid gyro notch parameters");

    gyro_enable_ = gyro_enable;
    notch_enable_ = notch_enable;
    acc_window_ = acc_window;
    if (gyro_enable_) {
      const double k = std::tan(3.14159265358979323846 * cutoff_hz / fs_hz);
      const double k2 = k * k;
      const double root2k = std::sqrt(2.0) * k;
      const double n = 1.0 / (1.0 + root2k + k2);
      for (int i = 0; i < 3; ++i) {
        biquad_[i].b0 = k2 * n;
        biquad_[i].b1 = 2.0 * biquad_[i].b0;
        biquad_[i].b2 = biquad_[i].b0;
        biquad_[i].a1 = 2.0 * (k2 - 1.0) * n;
        biquad_[i].a2 = (1.0 - root2k + k2) * n;
      }
    }
    if (notch_enable_) {
      // RBJ biquad notch. Unlike a low-pass, this preserves the low-frequency
      // motion and removes only the measured narrow-band motor tone.
      const double w0 = 2.0 * 3.14159265358979323846 * notch_hz / fs_hz;
      const double alpha = std::sin(w0) / (2.0 * notch_q);
      const double a0 = 1.0 + alpha;
      const double b0 = 1.0 / a0;
      const double b1 = -2.0 * std::cos(w0) / a0;
      const double b2 = b0;
      const double a1 = -2.0 * std::cos(w0) / a0;
      const double a2 = (1.0 - alpha) / a0;
      for (int i = 0; i < 3; ++i)
        notch_[i].configure(b0, b1, b2, a1, a2);
    }
    reset();
  }

  void reset() {
    for (int i = 0; i < 3; ++i) {
      biquad_[i].seeded = false;
      biquad_[i].x1 = biquad_[i].x2 = biquad_[i].y1 = biquad_[i].y2 = 0.0;
      notch_[i].seeded = false;
      notch_[i].x1 = notch_[i].x2 = notch_[i].y1 = notch_[i].y2 = 0.0;
      history_[i].clear();
    }
  }

  void filter(const double wm[3], const double am[3], double out_wm[3], double out_am[3]) {
    for (int i = 0; i < 3; ++i) {
      double gyro = notch_enable_ ? notch_[i].step(wm[i]) : wm[i];
      out_wm[i] = gyro_enable_ ? biquad_[i].step(gyro) : gyro;
      out_am[i] = acc_window_ ? median(i, am[i]) : am[i];
    }
  }

private:
  struct Biquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    bool seeded = false;

    void configure(double nb0, double nb1, double nb2, double na1, double na2) {
      b0 = nb0;
      b1 = nb1;
      b2 = nb2;
      a1 = na1;
      a2 = na2;
    }

    double step(double x) {
      if (!seeded) {
        seeded = true;
        x1 = x2 = y1 = y2 = x;
        return x;
      }
      const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      return y;
    }
  };

  double median(int axis, double x) {
    std::deque<double> &h = history_[axis];
    h.push_back(x);
    if (static_cast<int>(h.size()) > acc_window_)
      h.pop_front();
    std::deque<double> sorted = h;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
  }

  bool gyro_enable_ = false;
  bool notch_enable_ = false;
  int acc_window_ = 0;
  Biquad biquad_[3];
  Biquad notch_[3];
  std::deque<double> history_[3];
};

} // namespace ov_core

#endif

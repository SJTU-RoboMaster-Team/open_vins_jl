/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef OV_CORE_IMU_PREFILTER_H
#define OV_CORE_IMU_PREFILTER_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov_core {

/**
 * Stateless-config pre-filter for the RealSense T265 *united* IMU stream
 * (gyro+accel resampled together at fs = 199.94 Hz) feeding OpenVINS MSCKF.
 * Header-only, std C++14, no Eigen / no ROS, so it can be unit-tested
 * standalone (see ov_core/tests/imu_prefilter_main.cpp).
 *
 * Measured flight vibration (flight-* records, see docs/flight-log-analysis.md):
 *  - GYRO: narrow ~93 Hz tone, 31-95 Hz band RMS ~0.75 rad/s. The gyro is
 *    genuinely sampled at 199.94 Hz, so 93 Hz sits far below Nyquist and this
 *    per-axis 2nd-order Butterworth low-pass removes it. Useful signal
 *    bandwidth is <= ~3 Hz, leaving large cutoff headroom.
 *  - ACCEL: only structural peaks at 10.5 / 18.7 Hz; no in-band 93 Hz exists.
 *
 * DO NOT ADD AN ACCEL LOW-PASS. The T265 accelerometer natively samples at
 * 62.5 Hz, and the united stream merely interpolates those samples to
 * 199.94 Hz. Any 93 Hz mechanical tone on the accel therefore ALIASES to
 * |93 - 62.5| = 30.5 Hz *before* we ever see it; at 30.5 Hz it is
 * indistinguishable from genuine vehicle motion and lies above our ~3 Hz
 * useful band. That information was destroyed at sampling time and is
 * digitally unfixable post-sampling - a cutoff here would only smear real
 * dynamics. The accel path below is a causal median-of-N for isolated glitch
 * spikes (sensor dropouts / USB-frame artifacts), which is a robustness
 * filter, not a band filter.
 *
 * Gyro coefficients are computed once in configure() from the bilinear
 * transform of the analog Butterworth prototype WITH frequency prewarp
 * (k = tan(pi*fc/fs)), so the digital -3 dB point lands exactly at fc.
 * After reset(), each biquad seeds its state with the first sample, giving
 * no startup transient (the filter opens at the current bias point instead
 * of ramping from zero, which would corrupt the estimator at arm time).
 */
class ImuPreFilter {
public:
  ImuPreFilter() = default;

  /**
   * (Re)compute coefficients and clear all state.
   * gyro_lp_enable: false makes the gyro path a pure passthrough.
   * acc_median_window: 0 disables the accel median (passthrough);
   * otherwise the per-axis causal median over the last N samples.
   * Throws std::invalid_argument unless fs > 0, 0 < fc < fs/2 (checked when
   * the low-pass is enabled) and N is 0 or an odd number >= 3.
   */
  void configure(bool gyro_lp_enable, double gyro_fs_hz, double gyro_fc_hz, int acc_median_window) {
    if (!(acc_median_window == 0 || (acc_median_window >= 3 && acc_median_window % 2 == 1))) {
      throw std::invalid_argument("ImuPreFilter: acc_median_window must be 0 or odd >= 3, got " +
                                  std::to_string(acc_median_window));
    }
    if (gyro_lp_enable) {
      if (!(gyro_fs_hz > 0.0)) {
        throw std::invalid_argument("ImuPreFilter: gyro_fs_hz must be > 0, got " + std::to_string(gyro_fs_hz));
      }
      if (!(gyro_fc_hz > 0.0 && gyro_fc_hz < gyro_fs_hz / 2.0)) {
        throw std::invalid_argument("ImuPreFilter: gyro_fc_hz must satisfy 0 < fc < fs/2, got fc=" +
                                    std::to_string(gyro_fc_hz) + " fs=" + std::to_string(gyro_fs_hz));
      }
    }
    gyro_lp_ = gyro_lp_enable;
    acc_n_ = acc_median_window;
    if (gyro_lp_) {
      // Prewarped bilinear 2nd-order Butterworth low-pass (Q = 1/sqrt(2)):
      //   k = tan(pi*fc/fs), norm = 1/(1+sqrt(2)k+k^2)
      //   b = [k^2, 2k^2, k^2]*norm, a = [1, 2(k^2-1), 1-sqrt(2)k+k^2]*norm
      const double pi = 3.14159265358979323846;
      const double k = std::tan(pi * gyro_fc_hz / gyro_fs_hz);
      const double k2 = k * k;
      const double root2k = std::sqrt(2.0) * k;
      const double norm = 1.0 / (1.0 + root2k + k2);
      for (int i = 0; i < 3; i++) {
        bq_[i].b0 = k2 * norm;
        bq_[i].b1 = 2.0 * bq_[i].b0;
        bq_[i].b2 = bq_[i].b0;
        bq_[i].a1 = 2.0 * (k2 - 1.0) * norm;
        bq_[i].a2 = (1.0 - root2k + k2) * norm;
      }
    }
    reset();
  }

  /// Drop all filter history; the next sample re-seeds the gyro biquads.
  void reset() {
    for (int i = 0; i < 3; i++) {
      bq_[i].seeded = false;
      bq_[i].x1 = bq_[i].x2 = bq_[i].y1 = bq_[i].y2 = 0.0;
      hist_[i].clear();
    }
  }

  /// One IMU sample in, one filtered sample out (wm = gyro, am = accel).
  void filter(const double wm[3], const double am[3], double out_wm[3], double out_am[3]) {
    for (int i = 0; i < 3; i++) {
      out_wm[i] = gyro_lp_ ? bq_[i].step(wm[i]) : wm[i];
      out_am[i] = (acc_n_ > 0) ? median_window(i, am[i]) : am[i];
    }
  }

private:
  struct Biquad {
    // Direct form I: y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    bool seeded = false;
    double step(double x) {
      if (!seeded) {
        // Seed with the first sample: constant input -> constant output
        // (DC gain of this prototype is exactly 1), i.e. no startup transient.
        x1 = x2 = x;
        y1 = y2 = x;
        seeded = true;
        return x;
      }
      double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      return y;
    }
  };

  double median_window(int axis, double x) {
    std::deque<double> &h = hist_[axis];
    h.push_back(x);
    if ((int)h.size() > acc_n_) h.pop_front();
    std::vector<double> v(h.begin(), h.end());
    return stat_of_window(v);
  }

  static double stat_of_window(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  }

  bool gyro_lp_ = false;
  int acc_n_ = 0;
  Biquad bq_[3];
  std::deque<double> hist_[3];
};

} // namespace ov_core

#endif // OV_CORE_IMU_PREFILTER_H

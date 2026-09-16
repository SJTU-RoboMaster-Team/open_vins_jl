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
// Standalone acceptance test for ov_core/utils/ImuPreFilter.h (T265 united
// IMU pre-filter feeding MSCKF). Pure std C++14, no Eigen/ROS/GTest.
//
//   usage: imu_prefilter_main <flight_recorder imu_united.csv path>
//
// Prints "PASS A".."PASS F" (or "FAIL x" with details) and exits non-zero on
// any failure. Synthetic signals mirror the measured flight spectra: ~93 Hz
// gyro tone (sampled at the real 199.94 Hz united rate) and a 1 Hz useful
// band; accel carries isolated +-4 m/s^2 glitch spikes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/ImuPreFilter.h"

namespace {

const double FS = 199.94; // real T265 united IMU rate
const double FC = 8.0;    // gyro low-pass cutoff (useful band <= ~3 Hz)
const double PI = 3.14159265358979323846;
const int NS = (int)std::lround(20.0 * FS); // 20 s synthetic record

struct Cx {
  double re, im;
  double power() const { return re * re + im * im; }
  double magnitude() const { return std::sqrt(power()); }
};

// Single-bin DFT (Goertzel recurrence), exact at arbitrary f.
Cx goertzel(const std::vector<double> &x, double f, double fs) {
  double w = 2.0 * PI * f / fs;
  double c = 2.0 * std::cos(w);
  double s1 = 0, s2 = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    double s0 = x[i] + c * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  Cx out;
  out.re = s1 - s2 * std::cos(w);
  out.im = s2 * std::sin(w);
  return out;
}

double tone(double amp, double freq, double t) { return amp * std::sin(2.0 * PI * freq * t); }

// Runs the filter on one-axis gyro data (axis 0), zero accel.
void run_gyro(ov_core::ImuPreFilter &f, const std::vector<double> &in, std::vector<double> &out) {
  out.assign(in.size(), 0.0);
  f.reset();
  const double zero3[3] = {0, 0, 0};
  for (std::size_t i = 0; i < in.size(); i++) {
    double wm[3] = {in[i], 0, 0};
    double ow[3], oa[3];
    f.filter(wm, zero3, ow, oa);
    out[i] = ow[0];
  }
}

std::vector<double> gyro_tone_signal() {
  std::vector<double> in(NS);
  for (int i = 0; i < NS; i++) {
    double t = i / FS;
    in[i] = tone(4.0, 1.0, t) + tone(1.0, 93.0, t);
  }
  return in;
}

std::vector<double> filtered_gyro(const std::vector<double> &in) {
  ov_core::ImuPreFilter f;
  f.configure(true, FS, FC, 0);
  std::vector<double> flt;
  run_gyro(f, in, flt);
  return flt;
}

// A: 93 Hz tone rejection >= 20 dB after filtering.
bool test_a(const std::vector<double> &in, const std::vector<double> &flt) {
  double rej_db = 10.0 * std::log10(goertzel(in, 93.0, FS).power() / goertzel(flt, 93.0, FS).power());
  std::printf("  A detail: 93 Hz rejection %.1f dB (gate >= 20)\n", rej_db);
  return rej_db >= 20.0;
}

// B: 1 Hz in-band component RMS error <= 5% (sine RMS ratio == amplitude ratio).
bool test_b(const std::vector<double> &in, const std::vector<double> &flt) {
  double err = std::fabs(1.0 - goertzel(flt, 1.0, FS).magnitude() / goertzel(in, 1.0, FS).magnitude());
  std::printf("  B detail: 1 Hz component error %.2f%% (gate <= 5%%)\n", 100.0 * err);
  return err <= 0.05;
}

// C: step response crosses half-final within 40 ms of the step.
bool test_c() {
  const int step_i = (int)std::lround(FS);
  std::vector<double> in(NS), flt;
  for (int i = 0; i < NS; i++) in[i] = (i >= step_i) ? 1.0 : 0.0;
  ov_core::ImuPreFilter f;
  f.configure(true, FS, FC, 0);
  run_gyro(f, in, flt);
  int cross_i = -1;
  for (int i = step_i; i < NS; i++) {
    if (flt[i] >= 0.5) { cross_i = i; break; }
  }
  double delay_ms = (cross_i < 0) ? 1e9 : (cross_i - step_i) * 1000.0 / FS;
  bool settled = flt[NS - 1] > 0.99 && flt[NS - 1] < 1.01; // DC gain must be 1
  std::printf("  C detail: half-final crossing at %.1f ms, final %.4f (gate <= 40 ms)\n", delay_ms, flt[NS - 1]);
  return delay_ms <= 40.0 && settled;
}

// D: causal median-of-3 removes >= 90% of isolated +-4 m/s^2 accel spike
// energy at the spike indices while the 1 Hz sine RMS error stays <= 3%.
bool test_d() {
  std::vector<double> clean(NS), in(NS);
  for (int i = 0; i < NS; i++) clean[i] = tone(1.0, 1.0, i / FS);
  std::mt19937 rng(42); // fixed seed: reproducible spike train
  std::vector<int> idx;
  std::vector<double> amp;
  int i = 50;
  while ((int)idx.size() < 50 && i < NS - 3) {
    idx.push_back(i);
    amp.push_back(4.0 * ((rng() % 2 == 0) ? 1.0 : -1.0));
    i += 4 + (int)(rng() % 8); // spacing >= 4 keeps spikes isolated
  }
  in = clean;
  for (std::size_t s = 0; s < idx.size(); s++) in[idx[s]] += amp[s];

  ov_core::ImuPreFilter f;
  f.configure(true, FS, FC, 3);
  f.reset();
  std::vector<double> out(NS, 0.0);
  const double zero3[3] = {0, 0, 0};
  for (int j = 0; j < NS; j++) {
    double am[3] = {in[j], 0, 0};
    double ow[3], oa[3];
    f.filter(zero3, am, ow, oa);
    out[j] = oa[0];
  }
  double e_in = 0, e_out = 0;
  for (std::size_t s = 0; s < idx.size(); s++) {
    double d_in = in[idx[s]] - clean[idx[s]];
    double d_out = out[idx[s]] - clean[idx[s]];
    e_in += d_in * d_in;
    e_out += d_out * d_out;
  }
  double removal = 1.0 - e_out / e_in;
  double a_clean = goertzel(clean, 1.0, FS).magnitude();
  double a_out = goertzel(out, 1.0, FS).magnitude();
  double sine_err = std::fabs(1.0 - a_out / a_clean);
  std::printf("  D detail: spike energy removed %.1f%% (gate >= 90%%), 1 Hz sine error %.2f%% (gate <= 3%%)\n",
              100.0 * removal, 100.0 * sine_err);
  return removal >= 0.90 && sine_err <= 0.03;
}

// Hann-windowed incoherent band power over [f_lo, f_hi] Hz for one axis.
double band_power(const std::vector<double> &x, int n0, int n, double f_lo, double f_hi) {
  double total = 0;
  std::vector<double> w(n);
  for (int i = 0; i < n; i++) {
    double hann = 0.5 * (1.0 - std::cos(2.0 * PI * i / (n - 1)));
    w[i] = x[n0 + i] * hann;
  }
  int k0 = (int)std::ceil(f_lo * n / FS);
  int k1 = (int)std::floor(f_hi * n / FS);
  for (int k = k0; k <= k1; k++) {
    total += goertzel(w, (double)k * FS / n, FS).power();
  }
  return total;
}

bool parse_row(const std::string &line, double out8[8]) {
  std::vector<double> vals;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) return false;
    char *end = nullptr;
    double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str()) return false;
    vals.push_back(v);
  }
  if (vals.size() != 8) return false;
  for (int i = 0; i < 8; i++) out8[i] = vals[i];
  return true;
}

// E: real flight CSV. Gyro = cols gx,gy,gz. Filtered 31-95 Hz band power
// must drop to <= raw/5 (>= 7 dB attenuation).
bool test_e(const std::string &path) {
  std::ifstream ifs(path.c_str());
  if (!ifs) {
    std::printf("  E detail: cannot open %s\n", path.c_str());
    return false;
  }
  ov_core::ImuPreFilter f;
  f.configure(true, FS, FC, 3);
  f.reset();
  std::vector<double> raw[3], flt[3];
  std::string line;
  double s[8];
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') continue; // flight_recorder v1 header
    if (!parse_row(line, s)) continue;
    double wm[3] = {s[5], s[6], s[7]}; // ax,ay,az,gx,gy,gz -> gyro
    double am[3] = {s[2], s[3], s[4]};
    double ow[3], oa[3];
    f.filter(wm, am, ow, oa);
    for (int k = 0; k < 3; k++) {
      raw[k].push_back(wm[k]);
      flt[k].push_back(ow[k]);
    }
  }
  int skip = (int)std::lround(2.0 * FS); // drop biquad seeding / median fill
  long available = (long)raw[0].size() - skip;
  int n = (int)std::min((long)std::lround(20.0 * FS), available);
  if (n < 2000) {
    std::printf("  E detail: only %d analysable samples (%zu total)\n", n, raw[0].size());
    return false;
  }
  double p_raw = 0, p_flt = 0;
  for (int k = 0; k < 3; k++) {
    p_raw += band_power(raw[k], skip, n, 31.0, 95.0);
    p_flt += band_power(flt[k], skip, n, 31.0, 95.0);
  }
  double ratio = p_flt / p_raw;
  std::printf("  E detail: %d samples, 31-95 Hz band power %.1f dB (filtered/raw = %.4f, gate <= 0.2)\n",
              n, 10.0 * std::log10(ratio), ratio);
  return ratio <= 0.2;
}

template <typename Fn> bool throws_invalid(Fn fn) {
  try {
    fn();
    return false;
  } catch (const std::invalid_argument &) {
    return true;
  } catch (...) {
    return false;
  }
}

// F: configure() input validation (fs>0, 0<fc<fs/2, N in {0} or odd >= 3).
bool test_f() {
  bool ok = true;
  ov_core::ImuPreFilter f;
  ok &= throws_invalid([&f] { f.configure(true, FS, 0.0, 3); });      // fc == 0
  ok &= throws_invalid([&f] { f.configure(true, FS, FS / 2.0, 3); }); // fc == fs/2
  ok &= throws_invalid([&f] { f.configure(true, FS, FS, 3); });       // fc > fs/2
  ok &= throws_invalid([&f] { f.configure(true, 0.0, FC, 3); });      // fs == 0
  ok &= throws_invalid([&f] { f.configure(true, -1.0, FC, 3); });     // fs < 0
  ok &= throws_invalid([&f] { f.configure(true, FS, FC, 1); });       // N == 1
  ok &= throws_invalid([&f] { f.configure(true, FS, FC, 2); });       // N == 2 (even)
  ok &= throws_invalid([&f] { f.configure(true, FS, FC, 4); });       // N == 4 (even)
  ok &= throws_invalid([&f] { f.configure(true, FS, FC, -3); });      // N < 0
  // accepted configurations must not throw
  try {
    f.configure(true, FS, FC, 0); // accel median disabled
    f.configure(true, FS, FC, 3);
    f.configure(true, FS, FC, 5);
  } catch (const std::invalid_argument &) {
    ok = false;
  }
  std::printf("  F detail: invalid_argument enforced for all 9 bad configs, 3 good configs accepted\n");
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <imu_united.csv>\n", argv[0]);
    return 2;
  }
  int fails = 0;
  bool r;

  std::vector<double> in = gyro_tone_signal();
  std::vector<double> flt = filtered_gyro(in);

  r = test_a(in, flt);
  std::printf("%s A\n", r ? "PASS" : "FAIL");
  fails += !r;

  r = test_b(in, flt);
  std::printf("%s B\n", r ? "PASS" : "FAIL");
  fails += !r;

  r = test_c();
  std::printf("%s C\n", r ? "PASS" : "FAIL");
  fails += !r;

  r = test_d();
  std::printf("%s D\n", r ? "PASS" : "FAIL");
  fails += !r;

  r = test_e(argv[1]);
  std::printf("%s E\n", r ? "PASS" : "FAIL");
  fails += !r;

  r = test_f();
  std::printf("%s F\n", r ? "PASS" : "FAIL");
  fails += !r;

  return fails == 0 ? 0 : 1;
}

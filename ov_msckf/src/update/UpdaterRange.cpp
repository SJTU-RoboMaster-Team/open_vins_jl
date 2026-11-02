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
 */

#include "UpdaterRange.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace ov_msckf;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kPlaneCandidateCount = 5;

bool valid_options(const RangeAidOptions &options) {
  const Eigen::Matrix3d R_range_to_imu = options.R_range_to_imu;
  const bool finite = R_range_to_imu.allFinite() &&
                      options.p_range_in_imu.allFinite() &&
                      options.measurement_axis.allFinite();
  const bool proper_rotation =
      std::abs(R_range_to_imu.determinant() - 1.0) < 1e-6 &&
      (R_range_to_imu.transpose() * R_range_to_imu -
       Eigen::Matrix3d::Identity()).norm() < 1e-6;
  const bool valid_axis = finite && options.measurement_axis.norm() > 1e-9;
  const bool valid_limits =
      std::isfinite(options.min_range_m) &&
      std::isfinite(options.max_range_m) &&
      options.min_range_m > 0.0 &&
      options.max_range_m > options.min_range_m;
  const bool valid_noise = std::isfinite(options.noise_m) && options.noise_m > 0.0;
  const bool valid_age = std::isfinite(options.max_age_s) && options.max_age_s > 0.0;
  const bool valid_tilt =
      std::isfinite(options.max_tilt_deg) &&
      options.max_tilt_deg > 0.0 &&
      options.max_tilt_deg < 90.0;
  const bool valid_chi2 =
      std::isfinite(options.chi2_multiplier) &&
      options.chi2_multiplier > 0.0;
  return finite && proper_rotation && valid_axis && valid_limits && valid_noise &&
         valid_age && valid_tilt && valid_chi2;
}

} // namespace

RangePrediction ov_msckf::predict_planar_range(
    const Eigen::Vector3d &p_IinG, const Eigen::Vector3d &ray_G,
    double ground_plane_height_m, double max_tilt_rad) {
  RangePrediction prediction;
  if (!p_IinG.allFinite() || !ray_G.allFinite() ||
      !std::isfinite(ground_plane_height_m) ||
      !std::isfinite(max_tilt_rad) || max_tilt_rad <= 0.0 ||
      max_tilt_rad >= kPi / 2.0 || ray_G.norm() < 1e-9) {
    return prediction;
  }
  const Eigen::Vector3d ray = ray_G.normalized();
  prediction.ray_vertical = ray.z();
  prediction.tilt_rad =
      std::acos(std::max(-1.0, std::min(1.0, -ray.z())));
  if (prediction.tilt_rad > max_tilt_rad) {
    return prediction;
  }
  // The world frame is gravity aligned with +Z up.  A downward beam has a
  // negative Z component, so divide the positive height above the ground by
  // the downward vertical component.
  prediction.range_m = (p_IinG.z() - ground_plane_height_m) / -ray.z();
  prediction.valid = std::isfinite(prediction.range_m) && prediction.range_m > 0.0;
  return prediction;
}

double ov_msckf::estimate_ground_plane_height(
    const Eigen::Vector3d &p_IinG, const Eigen::Vector3d &ray_G,
    double measured_range_m) {
  if (!p_IinG.allFinite() || !ray_G.allFinite() ||
      !std::isfinite(measured_range_m) || measured_range_m <= 0.0 ||
      ray_G.norm() < 1e-9) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return p_IinG.z() + measured_range_m * ray_G.normalized().z();
}

UpdaterRange::UpdaterRange(const RangeAidOptions &options) : options_(options) {
  if (options_.enabled && !valid_options(options_)) {
    throw std::invalid_argument("range aid requires a proper mount, axis, limits, noise and gates");
  }
  if (options_.enabled) {
    options_.measurement_axis.normalize();
  }
}

RangeUpdateStatus UpdaterRange::reject(
    const char *reason, const ov_core::RangeData &measurement, double predicted,
    double innovation, double chi2, double tilt_deg) {
  ++diagnostics_.rejected;
  diagnostics_.last_range_stamp = measurement.timestamp;
  diagnostics_.last_range_m = measurement.range;
  diagnostics_.last_predicted_m = predicted;
  diagnostics_.last_innovation_m = innovation;
  diagnostics_.last_chi2 = chi2;
  diagnostics_.last_tilt_deg = tilt_deg;
  PRINT_INFO(YELLOW
             "[RANGE]: rejected reason=%s sensor_stamp_s=%.9f range_m=%.6f "
             "predicted_m=%.6f innovation_m=%.6f chi2=%.6f tilt_deg=%.3f "
             "accepted=%llu rejected=%llu\n" RESET,
             reason, measurement.timestamp, measurement.range, predicted,
             innovation, chi2, tilt_deg,
             static_cast<unsigned long long>(diagnostics_.accepted),
             static_cast<unsigned long long>(diagnostics_.rejected));
  return RangeUpdateStatus::kRejected;
}

RangeUpdateStatus UpdaterRange::try_update(
    const std::shared_ptr<State> &state,
    const ov_core::RangeData &measurement) {
  if (!options_.enabled || state == nullptr) {
    return RangeUpdateStatus::kDisabled;
  }

  const double age = state->_timestamp - measurement.timestamp;
  if (!std::isfinite(measurement.timestamp) ||
      !std::isfinite(measurement.range) ||
      !std::isfinite(age) || std::abs(age) > options_.max_age_s) {
    return reject("stale_or_invalid_timestamp", measurement);
  }

  const double driver_min =
      std::isfinite(measurement.min_range) ? measurement.min_range : 0.0;
  const double driver_max =
      std::isfinite(measurement.max_range)
          ? measurement.max_range
          : std::numeric_limits<double>::infinity();
  const double min_range = std::max(options_.min_range_m, driver_min);
  const double max_range = std::min(options_.max_range_m, driver_max);
  if (!(max_range > min_range) || measurement.range < min_range ||
      measurement.range > max_range) {
    return reject("range_out_of_bounds", measurement);
  }

  const Eigen::Matrix3d R_GtoI = state->_imu->Rot();
  const Eigen::Matrix3d R_ItoG = R_GtoI.transpose();
  const Eigen::Vector3d ray_G =
      R_ItoG * options_.R_range_to_imu * options_.measurement_axis;
  const Eigen::Vector3d p_RinG =
      state->_imu->pos() + R_ItoG * options_.p_range_in_imu;
  const Eigen::Vector3d ray_unit = ray_G.normalized();
  const double tilt_rad =
      std::acos(std::max(-1.0, std::min(1.0, -ray_unit.z())));
  const double tilt_limit_rad = options_.max_tilt_deg * kPi / 180.0;
  const double tilt_deg = tilt_rad * 180.0 / kPi;
  if (ray_unit.z() >= 0.0) {
    return reject("ray_not_downward", measurement, -1.0, 0.0, 0.0, tilt_deg);
  }
  if (tilt_rad > tilt_limit_rad) {
    return reject("tilt_limit", measurement, -1.0, 0.0, 0.0, tilt_deg);
  }

  if (!diagnostics_.plane_initialized) {
    const double candidate =
        estimate_ground_plane_height(p_RinG, ray_G, measurement.range);
    if (!std::isfinite(candidate)) {
      return reject("invalid_plane_candidate", measurement, -1.0, 0.0, 0.0,
                    tilt_deg);
    }
    if (last_plane_candidate_stamp_ >= 0.0 &&
        measurement.timestamp - last_plane_candidate_stamp_ >
            std::max(1.0, 5.0 * options_.max_age_s)) {
      plane_candidates_.clear();
    }
    plane_candidates_.push_back(candidate);
    last_plane_candidate_stamp_ = measurement.timestamp;
    while (plane_candidates_.size() > kPlaneCandidateCount) {
      plane_candidates_.pop_front();
    }
    const auto extrema =
        std::minmax_element(plane_candidates_.begin(), plane_candidates_.end());
    const double tolerance = std::max(0.05, 4.0 * options_.noise_m);
    if (*extrema.second - *extrema.first > tolerance) {
      const double latest = plane_candidates_.back();
      plane_candidates_.clear();
      plane_candidates_.push_back(latest);
    }
    if (plane_candidates_.size() < kPlaneCandidateCount) {
      ++diagnostics_.plane_pending;
      return RangeUpdateStatus::kPlanePending;
    }
    double sum = 0.0;
    for (double value : plane_candidates_) sum += value;
    ground_plane_height_m_ = sum / static_cast<double>(plane_candidates_.size());
    diagnostics_.plane_initialized = true;
    diagnostics_.ground_plane_height_m = ground_plane_height_m_;
    diagnostics_.last_range_stamp = measurement.timestamp;
    diagnostics_.last_range_m = measurement.range;
    diagnostics_.last_tilt_deg = tilt_deg;
    ++diagnostics_.accepted;
    PRINT_INFO(CYAN
               "[RANGE]: plane_initialized sensor_stamp_s=%.9f height_m=%.6f "
               "range_m=%.6f tilt_deg=%.3f samples=%zu\n" RESET,
               measurement.timestamp, ground_plane_height_m_, measurement.range,
               tilt_deg, plane_candidates_.size());
    return RangeUpdateStatus::kPlaneInitialized;
  }

  const RangePrediction geometry = predict_planar_range(
      p_RinG, ray_G, ground_plane_height_m_,
      tilt_limit_rad);
  if (!geometry.valid) {
    return reject("tilt_or_intersection_gate", measurement, -1.0, 0.0, 0.0,
                  tilt_deg);
  }
  const double min_configured = std::max(options_.min_range_m, driver_min);
  const double max_configured = std::min(options_.max_range_m, driver_max);
  if (geometry.range_m < min_configured || geometry.range_m > max_configured) {
    return reject("predicted_range_out_of_bounds", measurement,
                  geometry.range_m, 0.0, 0.0, tilt_deg);
  }

  const double innovation = measurement.range - geometry.range_m;
  std::vector<std::shared_ptr<ov_type::Type>> H_order{state->_imu->p()};
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 3);
  // h(p) = (plane_height - p_R.z) / ray.z, so dh/dp_z is negative.
  H(0, 2) = -1.0 / geometry.ray_vertical;
  Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, H_order);
  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(
      1, 1, options_.noise_m * options_.noise_m);
  const double innovation_variance = (H * P * H.transpose())(0, 0) + R(0, 0);
  if (!std::isfinite(innovation_variance) || innovation_variance <= 0.0) {
    return reject("invalid_innovation_covariance", measurement,
                  geometry.range_m, innovation, 0.0, tilt_deg);
  }
  const double chi2 = innovation * innovation / innovation_variance;
  const double chi2_limit = options_.chi2_multiplier * 3.841458820694124;
  if (!std::isfinite(chi2) || chi2 > chi2_limit) {
    return reject("innovation_chi2", measurement, geometry.range_m,
                  innovation, chi2, tilt_deg);
  }

  Eigen::VectorXd residual(1);
  residual << innovation;
  StateHelper::EKFUpdate(state, H_order, H, residual, R);
  diagnostics_.last_range_stamp = measurement.timestamp;
  diagnostics_.last_range_m = measurement.range;
  diagnostics_.last_predicted_m = geometry.range_m;
  diagnostics_.last_innovation_m = innovation;
  diagnostics_.last_chi2 = chi2;
  diagnostics_.last_tilt_deg = tilt_deg;
  const uint64_t accepted = ++diagnostics_.accepted;
  if (accepted == 1 || accepted % 100 == 0) {
    PRINT_INFO(CYAN
               "[RANGE]: accepted sensor_stamp_s=%.9f range_m=%.6f "
               "predicted_m=%.6f innovation_m=%.6f chi2=%.6f tilt_deg=%.3f "
               "accepted=%llu rejected=%llu height_m=%.6f\n" RESET,
               measurement.timestamp, measurement.range, geometry.range_m,
               innovation, chi2, tilt_deg,
               static_cast<unsigned long long>(accepted),
               static_cast<unsigned long long>(diagnostics_.rejected),
               state->_imu->pos().z());
  }
  return RangeUpdateStatus::kAccepted;
}

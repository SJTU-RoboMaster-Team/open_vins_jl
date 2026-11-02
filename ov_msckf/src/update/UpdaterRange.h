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

#ifndef OV_MSCKF_UPDATER_RANGE_H
#define OV_MSCKF_UPDATER_RANGE_H

#include <Eigen/Eigen>
#include <cstdint>
#include <deque>
#include <memory>

#include "utils/sensor_data.h"

namespace ov_msckf {

class State;

struct RangeAidOptions {
  bool enabled = false;
  /// Rotation from the range-sensor frame into the IMU frame.
  Eigen::Matrix3d R_range_to_imu = Eigen::Matrix3d::Identity();
  /// Range-sensor origin expressed in the IMU frame.
  Eigen::Vector3d p_range_in_imu = Eigen::Vector3d::Zero();
  /// Unit beam direction expressed in the range-sensor frame.
  Eigen::Vector3d measurement_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
  double min_range_m = 0.1;
  double max_range_m = 3.0;
  double noise_m = 0.05;
  double max_age_s = 0.10;
  double max_tilt_deg = 25.0;
  double chi2_multiplier = 1.0;
};

struct RangePrediction {
  bool valid = false;
  double range_m = 0.0;
  double ray_vertical = 0.0;
  double tilt_rad = 0.0;
};

/// Geometry-only helpers keep the ground-plane model independently testable.
RangePrediction predict_planar_range(const Eigen::Vector3d &p_IinG,
                                     const Eigen::Vector3d &ray_G,
                                     double ground_plane_height_m,
                                     double max_tilt_rad);

double estimate_ground_plane_height(const Eigen::Vector3d &p_IinG,
                                    const Eigen::Vector3d &ray_G,
                                    double measured_range_m);

struct RangeUpdateDiagnostics {
  uint64_t accepted = 0;
  uint64_t rejected = 0;
  uint64_t plane_pending = 0;
  double last_range_stamp = -1.0;
  double last_range_m = -1.0;
  double last_predicted_m = -1.0;
  double last_innovation_m = 0.0;
  double last_chi2 = 0.0;
  double last_tilt_deg = 0.0;
  bool plane_initialized = false;
  double ground_plane_height_m = 0.0;
};

enum class RangeUpdateStatus {
  kDisabled,
  kRejected,
  kPlanePending,
  kPlaneInitialized,
  kAccepted
};

/**
 * @brief Bounded downward range update against a gravity-aligned ground plane.
 *
 * The first stable set of samples freezes one horizontal ground plane below
 * the vehicle in the local +Z-up gravity-aligned VIO frame.  Later updates use
 * only the vertical position state; roll/pitch remain constrained by the IMU
 * and are gated by maximum tilt from the downward direction.  Range provides
 * height/scale information only, never horizontal absolute position.
 */
class UpdaterRange {
public:
  explicit UpdaterRange(const RangeAidOptions &options);

  RangeUpdateStatus try_update(const std::shared_ptr<State> &state,
                               const ov_core::RangeData &measurement);

  const RangeUpdateDiagnostics &diagnostics() const { return diagnostics_; }

protected:
  RangeUpdateStatus reject(const char *reason, const ov_core::RangeData &measurement,
                           double predicted = -1.0, double innovation = 0.0,
                           double chi2 = 0.0, double tilt_deg = 0.0);

  RangeAidOptions options_;
  RangeUpdateDiagnostics diagnostics_;
  double ground_plane_height_m_ = 0.0;
  std::deque<double> plane_candidates_;
  double last_plane_candidate_stamp_ = -1.0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_RANGE_H

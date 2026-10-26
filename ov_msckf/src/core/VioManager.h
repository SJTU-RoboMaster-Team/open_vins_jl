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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "VioManagerOptions.h"
#include "feat/FeatureDatabase.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class BoundedStatsLogger;
class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  using InstantStereoObservation = ov_core::FeatureDatabase::StereoObservation;

  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  ~VioManager();

  struct StatsLogDiagnostics { uint64_t written = 0, dropped = 0, truncated = 0, queued = 0; };
  StatsLogDiagnostics stats_log_diagnostics() const;

  /// Best-effort INFO record: fixed-copy enqueue only, safe under estimator locks.
  bool enqueue_info_diagnostic(const char *data, std::size_t size);

  /// Shutdown barrier: stop submitting measurements first; never hold the state
  /// mutex while waiting, since the initializer may need it to finish.
  void wait_for_initialization();

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /// ROS1's single camera worker may release its state lock for initialized
  /// frontend tracking. The lock is held again before any estimator mutation.
  void feed_measurement_camera(const ov_core::CameraData &message, std::unique_lock<std::mutex> &state_lock);

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// If we are initialized or not (requires at least one completed update)
  bool initialized() { return is_initialized_vio && timelastupdate != -1; }

  /// True once the estimator's initialization has been accepted, even before
  /// the first full feature update. During a stationary ground hold ZUPT can
  /// legitimately satisfy every frame, so `initialized()` stays false while
  /// high-rate odometry must still be publishable to downstream fusion.
  bool initialized_estimator() const { return is_initialized_vio; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Observations only. Caller holds get_state_mutex(); no estimator decisions
  /// depend on these counters. Gated counts are the updater's surviving features.
  struct UpdateDiagnostics {
    uint64_t zupt_accepted = 0, msckf_gated_total = 0, slam_gated_total = 0;
    size_t msckf_gated_last = 0, slam_gated_last = 0;
    double last_zupt_stamp = -1, last_msckf_stamp = -1, last_slam_stamp = -1;
    bool moved_since_zupt = false, last_zupt = false;
  };
  UpdateDiagnostics update_diagnostics() const {
    auto result = update_diagnostics_;
    result.moved_since_zupt = has_moved_since_zupt;
    result.last_zupt = did_zupt_update;
    return result;
  }

  /// Accessor for the immutable system parameters.
  const VioManagerOptions &get_params() const { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// ROS adapters must hold this across estimator reads/updates and fast
  /// propagation. The asynchronous initializer uses the same lock. In the
  /// synchronous initializer mode its joining caller already owns the lock.
  std::mutex &get_state_mutex() { return estimator_state_mtx; }

  /// Requires the caller's owned estimator lock; returns copied same-frame pixels only.
  std::vector<InstantStereoObservation> get_instant_stereo_observations(
      double timestamp, const std::unique_lock<std::mutex> &state_lock);

  /// Caller holds get_state_mutex(); debug images wait for a complete frontend.
  bool tracking_in_progress() const { return tracking_in_progress_; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image();

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Timestamp of the latest active-track sample, without copying its maps.
  double get_active_tracks_time() const { return active_tracks_time; }

  /// Output-only caches; caller holds the estimator state mutex. The default
  /// keeps non-ROS and ROS2 clients unchanged.
  void set_active_track_output(bool tracks, bool image) {
    if (!tracks && active_tracks_enabled_) {
      active_tracks_time = -1;
      active_image.release();
      active_tracks_posinG.clear();
      active_tracks_uvd.clear();
      active_feat_linsys_A.clear();
      active_feat_linsys_b.clear();
      active_feat_linsys_count.clear();
    }
    active_tracks_enabled_ = tracks;
    active_image_enabled_ = image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

protected:
  bool tracking_in_progress_ = false;
  bool active_tracks_enabled_ = true;
  bool active_image_enabled_ = true;

  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message, std::unique_lock<std::mutex> *state_lock = nullptr);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Shared with the ROS1 adapter and asynchronous initialization worker.
  std::mutex estimator_state_mtx;

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::unique_ptr<BoundedStatsLogger> stats_logger_;
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;
  std::thread initialization_thread_;

  // If we did a zero velocity update
  UpdateDiagnostics update_diagnostics_;
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H

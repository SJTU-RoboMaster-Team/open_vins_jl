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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include "bounded_stats_logger.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <sys/resource.h>
#include <time.h>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

// Assemble one by-value record without stdout, allocation, or any sink lock.
class StatsRecordBuilder {
public:
  void append(const char *format, ...) {
    if (truncated_) return;
    va_list args;
    va_start(args, format);
    const int count = vsnprintf(bytes_.data() + used_, bytes_.size() - used_, format, args);
    va_end(args);
    if (count < 0) { truncated_ = true; return; }
    if (static_cast<size_t>(count) >= bytes_.size() - used_) {
      truncated_ = true;
      bytes_.back() = '\n';
      used_ = bytes_.size();
    } else used_ += static_cast<size_t>(count);
  }
  const char *data() const { return bytes_.data(); }
  // Logger clamps to 4096 and counts oversized input, including truncated formatting.
  size_t size() const { return truncated_ ? bytes_.size() + 1 : used_; }
private:
  std::array<char, 4096> bytes_{};
  size_t used_ = 0;
  bool truncated_ = false;
};

// Process CPU and process rusage counters include work from concurrent threads.
struct PhaseMetrics {
  std::chrono::steady_clock::time_point wall;
  uint64_t thread_cpu_ns = 0;
  struct rusage thread_usage = {};
  struct rusage process_usage = {};
};

uint64_t current_thread_cpu_ns() {
  struct timespec ts;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

PhaseMetrics phase_metrics_now() {
  PhaseMetrics metrics;
  metrics.wall = std::chrono::steady_clock::now();
  metrics.thread_cpu_ns = current_thread_cpu_ns();
  getrusage(RUSAGE_THREAD, &metrics.thread_usage);
  getrusage(RUSAGE_SELF, &metrics.process_usage);
  return metrics;
}

double usage_cpu_ms(const struct rusage &usage) {
  return usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0 +
         usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
}

class SlowPhaseTimer {
public:
  SlowPhaseTimer(const char *phase, double sensor_stamp) : phase_(phase), sensor_stamp_(sensor_stamp), start_(phase_metrics_now()) {}
  ~SlowPhaseTimer() { finish(); }

  void finish() {
    if (finished_) return;
    finished_ = true;
    const PhaseMetrics end = phase_metrics_now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end.wall - start_.wall).count();
    if (wall_ms < 300.0) return;
    const double thread_cpu_ms = start_.thread_cpu_ns && end.thread_cpu_ns >= start_.thread_cpu_ns
                                     ? (end.thread_cpu_ns - start_.thread_cpu_ns) / 1e6
                                     : -1.0;
    PRINT_WARNING(YELLOW "[SLOW-PHASE] phase=%s sensor_stamp_s=%.9f wall_ms=%.1f thread_cpu_ms=%.1f "
                          "process_cpu_ms_all_threads=%.1f thread_minflt=%ld thread_majflt=%ld "
                          "thread_nvcsw=%ld thread_nivcsw=%ld process_minflt=%ld process_majflt=%ld "
                          "process_nvcsw=%ld process_nivcsw=%ld\n" RESET,
                  phase_, sensor_stamp_, wall_ms, thread_cpu_ms,
                  usage_cpu_ms(end.process_usage) - usage_cpu_ms(start_.process_usage),
                  end.thread_usage.ru_minflt - start_.thread_usage.ru_minflt,
                  end.thread_usage.ru_majflt - start_.thread_usage.ru_majflt,
                  end.thread_usage.ru_nvcsw - start_.thread_usage.ru_nvcsw,
                  end.thread_usage.ru_nivcsw - start_.thread_usage.ru_nivcsw,
                  end.process_usage.ru_minflt - start_.process_usage.ru_minflt,
                  end.process_usage.ru_majflt - start_.process_usage.ru_majflt,
                  end.process_usage.ru_nvcsw - start_.process_usage.ru_nvcsw,
                  end.process_usage.ru_nivcsw - start_.process_usage.ru_nivcsw);
  }

  SlowPhaseTimer(const SlowPhaseTimer &) = delete;
  SlowPhaseTimer &operator=(const SlowPhaseTimer &) = delete;

private:
  const char *phase_;
  double sensor_stamp_;
  PhaseMetrics start_;
  bool finished_ = false;
};

} // namespace

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity,
                                                        params.zupt_require_visual_stationary);
  }
  stats_logger_.reset(new BoundedStatsLogger());
}

VioManager::StatsLogDiagnostics VioManager::stats_log_diagnostics() const {
  StatsLogDiagnostics result;
  if (!stats_logger_) return result;
  const auto counters = stats_logger_->counters();
  result.written = counters.written;
  result.dropped = counters.dropped_full + counters.dropped_contended + counters.dropped_sink;
  result.truncated = counters.truncated;
  result.queued = counters.queued;
  return result;
}

bool VioManager::enqueue_info_diagnostic(const char *data, std::size_t size) {
  if (static_cast<int>(Printer::current_print_level) > static_cast<int>(Printer::PrintLevel::INFO)) return false;
  return stats_logger_ && stats_logger_->try_enqueue(data, size);
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity,
                                                          params.zupt_require_visual_stationary);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {
  update_diagnostics_.msckf_gated_last = 0;
  update_diagnostics_.slam_gated_last = 0;

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  {
    SlowPhaseTimer phase("camera_preprocess_downsample", message.timestamp);
    for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
      cv::Mat img = message.images.at(i);
      cv::Mat mask = message.masks.at(i);
      cv::Mat img_temp, mask_temp;
      cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
      message.images.at(i) = img_temp;
      cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
      message.masks.at(i) = mask_temp;
    }
  }

  // Perform our feature tracking!
  {
    SlowPhaseTimer phase("camera_feature_tracking", message.timestamp);
    trackFEATS->feed_new_camera(message);
  }

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    SlowPhaseTimer phase("camera_aruco_tracking", message.timestamp);
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    SlowPhaseTimer phase("camera_zupt_try_and_cleanup", message.timestamp);
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
      if (did_zupt_update) {
        ++update_diagnostics_.zupt_accepted;
        update_diagnostics_.last_zupt_stamp = message.timestamp;
      }
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      // A successful ZUPT is a complete estimator update even though it
      // returns before do_feature_propagate_update(). Mark the first such
      // update so initialized() can release odometry/pose publishers during
      // a stationary start; otherwise timelastupdate stays at -1 until the
      // vehicle moves and downstream consumers never receive the pose needed
      // to start it.
      if (timelastupdate == -1) {
        timelastupdate = message.timestamp;
      }
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    {
      SlowPhaseTimer phase("camera_initialization", message.timestamp);
      is_initialized_vio = try_to_initialize(message);
    }
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    SlowPhaseTimer phase("estimator_propagation_and_clone", message.timestamp);
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  SlowPhaseTimer msckf_phase("estimator_msckf_feature_selection_and_update", message.timestamp);
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  updaterMSCKF->update(state, featsup_MSCKF);
  update_diagnostics_.msckf_gated_last = featsup_MSCKF.size();
  update_diagnostics_.msckf_gated_total += featsup_MSCKF.size();
  if (!featsup_MSCKF.empty()) update_diagnostics_.last_msckf_stamp = message.timestamp;
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();
  msckf_phase.finish();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  SlowPhaseTimer slam_phase("estimator_slam_update", message.timestamp);
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  update_diagnostics_.slam_gated_last = feats_slam_UPDATE.size();
  update_diagnostics_.slam_gated_total += feats_slam_UPDATE.size();
  if (!feats_slam_UPDATE.empty()) update_diagnostics_.last_slam_stamp = message.timestamp;
  rT5 = boost::posix_time::microsec_clock::local_time();
  slam_phase.finish();
  SlowPhaseTimer delayed_init_phase("estimator_slam_delayed_init", message.timestamp);
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();
  delayed_init_phase.finish();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  SlowPhaseTimer marginalization_phase("estimator_retriangulation_cleanup_marginalization", message.timestamp);
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  rT7 = boost::posix_time::microsec_clock::local_time();
  marginalization_phase.finish();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  SlowPhaseTimer reporting_phase("estimator_final_stats_and_output", message.timestamp);
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  if (static_cast<int>(Printer::current_print_level) <= static_cast<int>(Printer::PrintLevel::DEBUG)) {
    std::stringstream ss;
    ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
    for (const auto &id : message.sensor_ids) {
      ss << " " << id;
    }
    ss << ")" << std::endl;
    PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());
  }

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // State reads and formatting remain serialized with estimation. Sink I/O runs
  // on the owned logger worker and can never hold the estimator/camera locks.
  if (static_cast<int>(Printer::current_print_level) <= static_cast<int>(Printer::PrintLevel::INFO)) {
    StatsRecordBuilder stats;
    stats.append("[VIO-STATS] sensor_stamp_s=%.9f\n", message.timestamp);
    // Debug, print our current state
    stats.append("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
               state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
               state->_imu->pos()(2), distance);
    stats.append("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
               state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

    // Debug for camera imu offset
    if (state->_options.do_calib_camera_timeoffset) {
      stats.append("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
    }

    // Debug for camera intrinsics
    if (state->_options.do_calib_camera_intrinsics) {
      for (int i = 0; i < state->_options.num_cameras; i++) {
        std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
        stats.append("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                   calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
      }
    }

    // Debug for camera extrinsics
    if (state->_options.do_calib_camera_pose) {
      for (int i = 0; i < state->_options.num_cameras; i++) {
        std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
        stats.append("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                   calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
      }
    }

    // Debug for imu intrinsics
    if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
      stats.append("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
                 state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
    }
    if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
      stats.append("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
                 state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
    }
    if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
      stats.append("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
                 state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
                 state->_calib_imu_dw->value()(5));
      stats.append("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
                 state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
                 state->_calib_imu_da->value()(5));
    }
    if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
      stats.append("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
                 state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
                 state->_calib_imu_dw->value()(5));
      stats.append("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
                 state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
                 state->_calib_imu_da->value()(5));
    }
    if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
      stats.append("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
                 state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
                 state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
                 state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
    }
    enqueue_info_diagnostic(stats.data(), stats.size());
  }
}

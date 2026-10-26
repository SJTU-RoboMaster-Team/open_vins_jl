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

#include "ROS1Visualizer.h"

#include "core/VioManager.h"
#include "ros/ROSVisualizerHelper.h"
#include "sim/Simulator.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/camera_sync.h"
#include "utils/camera_backlog.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

ROS1Visualizer::ROS1Visualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim)
    : _nh(nh), _app(app), _sim(sim) {

  const auto &options = _app->get_params();
  async_subs_ = options.use_multi_threading_subs;
  async_pubs_ = options.use_multi_threading_pubs;
  expected_cameras_ = options.state_options.num_cameras;
  camera_min_interval_s_ = 1.0 / options.track_frequency;
  if (options.use_mask)
    camera_masks_ = options.masks;
  if (camera_masks_.empty()) {
    zero_camera_masks_.resize(expected_cameras_);
    for (size_t cam = 0; cam < expected_cameras_ && cam < options.camera_intrinsics.size(); ++cam) {
      const auto &intrinsics = options.camera_intrinsics.at(cam);
      zero_camera_masks_.at(cam) = cv::Mat::zeros(intrinsics->h(), intrinsics->w(), CV_8UC1);
    }
  }
  nh->param<int>("camera_queue_max_messages", camera_queue_max_messages_, std::max<int>(16, 4 * expected_cameras_));
  nh->param<double>("camera_queue_max_span_s", camera_queue_max_span_s_, 0.15);
  nh->param<double>("camera_queue_max_age_s", camera_queue_max_age_s_, 0.15);
  nh->param<bool>("camera_tracking_outside_state_lock", camera_tracking_outside_state_lock_, true);
  nh->param<bool>("demand_driven_loop_output", demand_driven_loop_output_, true);
  nh->param<int>("camera_subscriber_queue_size", camera_subscriber_queue_size_, 2);
  if (camera_subscriber_queue_size_ < 1 || camera_subscriber_queue_size_ > 100)
    throw std::invalid_argument("camera_subscriber_queue_size must be in [1, 100]");
  if (camera_queue_max_messages_ < static_cast<int>(expected_cameras_) ||
      !std::isfinite(camera_queue_max_span_s_) || camera_queue_max_span_s_ <= 0 ||
      !std::isfinite(camera_queue_max_age_s_) || camera_queue_max_age_s_ < 0)
    throw std::invalid_argument("camera queue limits must hold one group and have a finite positive span");
  PRINT_INFO("ROS1 camera transport: max_age_s=%.3f tracking_outside_state_lock=%d demand_driven_loop_output=%d\n",
             camera_queue_max_age_s_, static_cast<int>(camera_tracking_outside_state_lock_),
             static_cast<int>(demand_driven_loop_output_));

  // Output-only work is optional. Real flights consume odometry topics and do
  // not need the estimator's diagnostic TF tree.
  nh->param<bool>("publish_global_to_imu_tf", publish_global2imu_tf, true);
  nh->param<bool>("publish_calibration_tf", publish_calibration_tf, true);
  nh->param<bool>("path_history_without_subscribers", path_history_without_subscribers_, true);
  if (publish_global2imu_tf || publish_calibration_tf)
    mTfBr = std::make_shared<tf::TransformBroadcaster>();
  PRINT_INFO("ROS1 visual outputs: global_to_imu_tf=%d calibration_tf=%d path_history_without_subscribers=%d\n",
             static_cast<int>(publish_global2imu_tf), static_cast<int>(publish_calibration_tf),
             static_cast<int>(path_history_without_subscribers_));

  nh->param<double>("camera_sync_tolerance_s", camera_sync_tolerance_s_, 0.0);
  if (!std::isfinite(camera_sync_tolerance_s_) || camera_sync_tolerance_s_ < 0.0) {
    PRINT_ERROR("Invalid camera_sync_tolerance_s; disabling camera bundling\n");
    camera_sync_tolerance_s_ = 0.0;
  }
  if (camera_sync_tolerance_s_ > 0.0) {
    PRINT_INFO("Near-synchronous camera bundling enabled (tolerance %.3f s)\n", camera_sync_tolerance_s_);
  }

  PRINT_INFO("ROS1 camera transport: max_messages=%d max_span_s=%.3f subscriber_queue=%d persistent_worker=%d\n",
             camera_queue_max_messages_, camera_queue_max_span_s_, camera_subscriber_queue_size_, static_cast<int>(async_subs_));

  // Create image transport
  image_transport::ImageTransport it(*_nh);

  // Setup pose and path publisher
  pub_poseimu = nh->advertise<geometry_msgs::PoseWithCovarianceStamped>("poseimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_poseimu.getTopic().c_str());
  pub_odomimu = nh->advertise<nav_msgs::Odometry>("odomimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_odomimu.getTopic().c_str());
  pub_pathimu = nh->advertise<nav_msgs::Path>("pathimu", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathimu.getTopic().c_str());

  // 3D points publishing
  pub_points_msckf = nh->advertise<sensor_msgs::PointCloud2>("points_msckf", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf.getTopic().c_str());
  pub_points_slam = nh->advertise<sensor_msgs::PointCloud2>("points_slam", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_msckf.getTopic().c_str());
  pub_points_aruco = nh->advertise<sensor_msgs::PointCloud2>("points_aruco", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_aruco.getTopic().c_str());
  pub_points_sim = nh->advertise<sensor_msgs::PointCloud2>("points_sim", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_points_sim.getTopic().c_str());

  // Our tracking image
  it_pub_tracks = it.advertise("trackhist", 2);
  PRINT_DEBUG("Publishing: %s\n", it_pub_tracks.getTopic().c_str());

  // Groundtruth publishers
  pub_posegt = nh->advertise<geometry_msgs::PoseStamped>("posegt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_posegt.getTopic().c_str());
  pub_pathgt = nh->advertise<nav_msgs::Path>("pathgt", 2);
  PRINT_DEBUG("Publishing: %s\n", pub_pathgt.getTopic().c_str());

  // Loop closure publishers
  pub_loop_pose = nh->advertise<nav_msgs::Odometry>("loop_pose", 2);
  pub_loop_point = nh->advertise<sensor_msgs::PointCloud>("loop_feats", 2);
  {
    // Every camera publishes its own IMU->camera extrinsic and intrinsics so
    // the pose graph can build keyframes for more than the front camera.
    // Camera 0 keeps the historical topic names.
    const size_t loop_cam_count = std::max<size_t>(1, _app->get_params().camera_intrinsics.size());
    pub_loop_extrinsic.resize(loop_cam_count);
    pub_loop_intrinsics.resize(loop_cam_count);
    for (size_t c = 0; c < loop_cam_count; ++c) {
      const std::string suffix = (c == 0) ? std::string() : ("_cam" + std::to_string(c));
      pub_loop_extrinsic.at(c) = nh->advertise<nav_msgs::Odometry>("loop_extrinsic" + suffix, 2);
      pub_loop_intrinsics.at(c) = nh->advertise<sensor_msgs::CameraInfo>("loop_intrinsics" + suffix, 2);
      PRINT_DEBUG("Publishing: %s\n", pub_loop_intrinsics.at(c).getTopic().c_str());
    }
  }
  it_pub_loop_img_depth = it.advertise("loop_depth", 2);
  it_pub_loop_img_depth_color = it.advertise("loop_depth_colored", 2);

  // Optional vibration handling for T265 replays. The united stream is about
  // 200 Hz; use a causal gyro low-pass and only a short accel median so that
  // isolated transport glitches are removed without filtering the accel's
  // already-interpolated dynamics. Parameters are private to this node.
  bool gyro_lp_enable = true;
  double gyro_lp_fc_hz = 8.0;
  bool gyro_notch_enable = false;
  double gyro_notch_hz = 93.0;
  double gyro_notch_q = 8.0;
  int acc_median_window = 3;
  double imu_filter_fs_hz = 199.94;
  nh->param<bool>("imu_pre_filter_enable", imu_pre_filter_enable_, false);
  nh->param<bool>("imu_gyro_lp_enable", gyro_lp_enable, true);
  nh->param<double>("imu_gyro_lp_fc_hz", gyro_lp_fc_hz, 8.0);
  nh->param<bool>("imu_gyro_notch_enable", gyro_notch_enable, false);
  nh->param<double>("imu_gyro_notch_hz", gyro_notch_hz, 93.0);
  nh->param<double>("imu_gyro_notch_q", gyro_notch_q, 8.0);
  nh->param<int>("imu_acc_median_window", acc_median_window, 3);
  nh->param<double>("imu_filter_fs_hz", imu_filter_fs_hz, 199.94);
  try {
    imu_pre_filter_.configure(gyro_lp_enable, imu_filter_fs_hz, gyro_lp_fc_hz, acc_median_window,
                               gyro_notch_enable, gyro_notch_hz, gyro_notch_q);
  } catch (const std::exception &e) {
    imu_pre_filter_enable_ = false;
    PRINT_ERROR(RED "IMU pre-filter disabled: %s\n" RESET, e.what());
  }
  PRINT_INFO(REDPURPLE "IMU pre-filter: enable=%d gyro_lp=%d fc=%.1fHz notch=%d@%.1fHz Q=%.2f acc_median=%d fs=%.2fHz\n" RESET,
             static_cast<int>(imu_pre_filter_enable_), static_cast<int>(gyro_lp_enable), gyro_lp_fc_hz,
             static_cast<int>(gyro_notch_enable), gyro_notch_hz, gyro_notch_q, acc_median_window, imu_filter_fs_hz);

  // Load groundtruth if we have it and are not doing simulation
  // NOTE: needs to be a csv ASL format file
  if (nh->hasParam("path_gt") && _sim == nullptr) {
    std::string path_to_gt;
    nh->param<std::string>("path_gt", path_to_gt, "");
    if (!path_to_gt.empty()) {
      DatasetReader::load_gt_file(path_to_gt, gt_states);
      PRINT_DEBUG("gt file path is: %s\n", path_to_gt.c_str());
    }
  }

  // Load if we should save the total state to file
  // If so, then open the file and create folders as needed
  nh->param<bool>("save_total_state", save_total_state, false);
  if (save_total_state) {

    // files we will open
    std::string filepath_est, filepath_std, filepath_gt;
    nh->param<std::string>("filepath_est", filepath_est, "state_estimate.txt");
    nh->param<std::string>("filepath_std", filepath_std, "state_deviation.txt");
    nh->param<std::string>("filepath_gt", filepath_gt, "state_groundtruth.txt");

    // If it exists, then delete it
    if (boost::filesystem::exists(filepath_est))
      boost::filesystem::remove(filepath_est);
    if (boost::filesystem::exists(filepath_std))
      boost::filesystem::remove(filepath_std);

    // Create folder path to this location if not exists
    boost::filesystem::create_directories(boost::filesystem::path(filepath_est.c_str()).parent_path());
    boost::filesystem::create_directories(boost::filesystem::path(filepath_std.c_str()).parent_path());

    // Open the files
    of_state_est.open(filepath_est.c_str());
    of_state_std.open(filepath_std.c_str());
    of_state_est << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                 << std::endl;
    of_state_std << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                 << std::endl;

    // Groundtruth if we are simulating
    if (_sim != nullptr) {
      if (boost::filesystem::exists(filepath_gt))
        boost::filesystem::remove(filepath_gt);
      boost::filesystem::create_directories(boost::filesystem::path(filepath_gt.c_str()).parent_path());
      of_state_gt.open(filepath_gt.c_str());
      of_state_gt << "# timestamp(s) q p v bg ba cam_imu_dt num_cam cam0_k cam0_d cam0_rot cam0_trans ... imu_model dw da tg wtoI atoI etc"
                  << std::endl;
    }
  }

  // Start thread for the image publishing
  if (async_pubs_) {
    image_thread_ = std::thread([this] {
      ros::WallRate loop_rate(20);
      while (!stopping_.load() && ros::ok()) {
        if (it_pub_tracks.getNumSubscribers() != 0) {
          const PhaseSnapshot lock_start = phase_snapshot();
          std::lock_guard<std::mutex> state_lock(_app->get_state_mutex());
          record_timing(image_state_lock_wait_timing_, "image_state_lock_wait", lock_start,
                        _app->get_state()->_timestamp);
          if (!_app->tracking_in_progress())
            publish_images();
        }
        loop_rate.sleep();
      }
    });
  }

  // One owned worker replaces construction/join of a thread on every IMU.
  // Wake on ingress, never poll or retain a callback's stack measurement.
  if (async_subs_) {
    update_thread_ = std::thread([this] {
      uint64_t handled = 0;
      while (!stopping_.load()) {
        {
          std::unique_lock<std::mutex> lock(camera_queue_mtx);
          camera_queue_cv_.wait(lock, [this, &handled] {
            return stopping_.load() || camera_queue_generation_ != handled;
          });
          if (stopping_.load())
            return;
          handled = camera_queue_generation_;
        }
        process_camera_queue();
      }
    });
  }

  pub_recording_diagnostics_ = nh->advertise<diagnostic_msgs::DiagnosticArray>("diagnostics", 1);
  recording_diagnostics_timer_ = nh->createWallTimer(
      ros::WallDuration(0.5), &ROS1Visualizer::publish_recording_diagnostics, this);
  timing_diagnostics_timer_ = nh->createWallTimer(
      ros::WallDuration(5.0), &ROS1Visualizer::report_timing_diagnostics, this);
}

ROS1Visualizer::~ROS1Visualizer() {
  stop_workers();
  _app->wait_for_initialization();
}

void ROS1Visualizer::stop_workers() {
  std::lock_guard<std::mutex> ownership_lock(worker_ownership_mtx_);
  if (workers_stopped_)
    return;
  {
    std::lock_guard<std::mutex> queue_lock(camera_queue_mtx);
    stopping_.store(true);
  }
  camera_queue_cv_.notify_all();
  timing_diagnostics_timer_.stop();
  recording_diagnostics_timer_.stop();
  if (update_thread_.joinable())
    update_thread_.join();
  if (image_thread_.joinable())
    image_thread_.join();
  workers_stopped_ = true;
}

uint64_t ROS1Visualizer::steady_now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

void ROS1Visualizer::update_timing_max(std::atomic<uint64_t> &target, uint64_t value) {
  uint64_t current = target.load(std::memory_order_relaxed);
  while (current < value &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
}

ROS1Visualizer::PhaseSnapshot ROS1Visualizer::phase_snapshot() {
  PhaseSnapshot snapshot;
  snapshot.steady_ns = steady_now_ns();
  struct rusage usage;
  if (getrusage(RUSAGE_THREAD, &usage) == 0) {
    // getrusage already provides the thread CPU accounting. Avoid a second
    // CLOCK_THREAD_CPUTIME_ID syscall on every hot-path snapshot.
    snapshot.cpu_ns =
        static_cast<uint64_t>(usage.ru_utime.tv_sec) * 1000000000ULL +
        static_cast<uint64_t>(usage.ru_utime.tv_usec) * 1000ULL +
        static_cast<uint64_t>(usage.ru_stime.tv_sec) * 1000000000ULL +
        static_cast<uint64_t>(usage.ru_stime.tv_usec) * 1000ULL;
    snapshot.minor_faults = usage.ru_minflt;
    snapshot.major_faults = usage.ru_majflt;
    snapshot.voluntary_switches = usage.ru_nvcsw;
    snapshot.involuntary_switches = usage.ru_nivcsw;
  }
  return snapshot;
}

ROS1Visualizer::PhaseSnapshot ROS1Visualizer::record_timing(
    TimingStats &stats, const char *phase, const PhaseSnapshot &start,
    double sensor_stamp) {
  const PhaseSnapshot end = phase_snapshot();
  const uint64_t elapsed_ns = end.steady_ns >= start.steady_ns ? end.steady_ns - start.steady_ns : 0;
  stats.count.fetch_add(1, std::memory_order_relaxed);
  update_timing_max(stats.max_ns, elapsed_ns);
  if (elapsed_ns > 100000000ULL) {
    ROS_WARN_STREAM("OpenVINS slow phase=" << phase
                    << " wall_unix_s=" << ros::WallTime::now().toSec()
                    << " steady_ns=" << end.steady_ns
                    << " sensor_stamp_s=" << sensor_stamp
                    << " elapsed_ms=" << elapsed_ns / 1e6
                    << " cpu_ms=" << (end.cpu_ns >= start.cpu_ns ? end.cpu_ns - start.cpu_ns : 0) / 1e6
                    << " minflt=" << end.minor_faults - start.minor_faults
                    << " majflt=" << end.major_faults - start.major_faults
                    << " nvcsw=" << end.voluntary_switches - start.voluntary_switches
                    << " nivcsw=" << end.involuntary_switches - start.involuntary_switches);
  }
  return end;
}

void ROS1Visualizer::report_timing_diagnostics(const ros::WallTimerEvent &) {
  const auto take_max_ms = [](std::atomic<uint64_t> &value) {
    return static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / 1e6;
  };
  const uint64_t imu_count = imu_callback_count_.exchange(0, std::memory_order_relaxed);
  const double imu_gap_ms = take_max_ms(imu_max_receipt_gap_ns_);
  const uint64_t imu_feed_count = imu_feed_timing_.count.exchange(0, std::memory_order_relaxed);
  const double imu_feed_ms = take_max_ms(imu_feed_timing_.max_ns);
  const uint64_t imu_vis_count = imu_visualize_timing_.count.exchange(0, std::memory_order_relaxed);
  const double imu_vis_ms = take_max_ms(imu_visualize_timing_.max_ns);
  const uint64_t cam_feed_count = camera_feed_timing_.count.exchange(0, std::memory_order_relaxed);
  const double cam_feed_ms = take_max_ms(camera_feed_timing_.max_ns);
  const uint64_t cam_vis_count = camera_visualize_timing_.count.exchange(0, std::memory_order_relaxed);
  const double cam_vis_ms = take_max_ms(camera_visualize_timing_.max_ns);
  const uint64_t cam_lock_count = camera_queue_lock_timing_.count.exchange(0, std::memory_order_relaxed);
  const double cam_lock_ms = take_max_ms(camera_queue_lock_timing_.max_ns);
  const double imu_state_wait_ms = take_max_ms(imu_state_lock_wait_timing_.max_ns);
  const double cam_state_wait_ms = take_max_ms(camera_state_lock_wait_timing_.max_ns);
  const double image_state_wait_ms = take_max_ms(image_state_lock_wait_timing_.max_ns);
  const uint64_t invalid_odom_count = invalid_fast_odom_count_.exchange(0, std::memory_order_relaxed);
  const auto stats = _app->stats_log_diagnostics();
  size_t pending;
  {
    std::lock_guard<std::mutex> lock(camera_queue_mtx);
    pending = camera_queue.size();
  }
  ROS_INFO_STREAM("OpenVINS wall diagnostics window_s=5"
                  << " imu_callbacks=" << imu_count << " imu_max_receipt_gap_ms=" << imu_gap_ms
                  << " imu_feed_count=" << imu_feed_count << " imu_feed_max_ms=" << imu_feed_ms
                  << " imu_visualize_count=" << imu_vis_count << " imu_visualize_max_ms=" << imu_vis_ms
                  << " camera_feed_count=" << cam_feed_count << " camera_feed_max_ms=" << cam_feed_ms
                  << " camera_visualize_count=" << cam_vis_count << " camera_visualize_max_ms=" << cam_vis_ms
                  << " camera_queue_lock_count=" << cam_lock_count << " camera_queue_lock_max_ms=" << cam_lock_ms
                  << " imu_state_lock_wait_max_ms=" << imu_state_wait_ms
                  << " camera_state_lock_wait_max_ms=" << cam_state_wait_ms
                  << " image_state_lock_wait_max_ms=" << image_state_wait_ms
                  << " invalid_fast_odom=" << invalid_odom_count
                  << " camera_queue_pending=" << pending
                  << " camera_queue_peak=" << camera_queue_peak_.exchange(0)
                  << " camera_queue_dropped=" << camera_queue_dropped_.load()
                  << " camera_queue_expired=" << camera_queue_expired_.load()
                  << " camera_update_age_max_ms=" << take_max_ms(camera_update_age_max_ns_)
                  << " stats_log_written=" << stats.written << " stats_log_dropped=" << stats.dropped
                  << " stats_log_truncated=" << stats.truncated << " stats_log_queued=" << stats.queued);
}

void ROS1Visualizer::publish_recording_diagnostics(const ros::WallTimerEvent &) {
  if (stopping_.load() || pub_recording_diagnostics_.getNumSubscribers() == 0)
    return;
  diagnostic_msgs::DiagnosticArray message;
  message.header.stamp = ros::Time::now();
  diagnostic_msgs::DiagnosticStatus status;
  status.name = "openvins/estimator_and_transport";
  status.hardware_id = "seeker4";
  status.message = "observations; gated feature counts are not independent accuracy evidence";
  auto add = [&status](const std::string &key, const auto &value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    diagnostic_msgs::KeyValue pair;
    pair.key = key; pair.value = text.str(); status.values.push_back(std::move(pair));
  };
  // This observer never waits on the hot path. Queue/state snapshots may be
  // from adjacent instants and explicitly flag a skipped state snapshot.
  {
    std::unique_lock<std::mutex> lock(camera_queue_mtx, std::try_to_lock);
    add("queue_snapshot_available", lock.owns_lock());
    if (lock.owns_lock()) {
      add("queue_images", camera_queue.size());
      add("latest_imu_stamp_s", latest_imu_timestamp_);
      add("queue_oldest_stamp_s", camera_queue.empty() ? -1.0 : camera_queue.front().timestamp);
      add("queue_newest_stamp_s", camera_queue.empty() ? -1.0 : camera_queue.back().timestamp);
    }
  }
  add("queue_dropped_images", camera_queue_dropped_.load());
  add("queue_expired_images", camera_queue_expired_.load());
  add("camera_queue_max_age_s", camera_queue_max_age_s_);
  {
    std::unique_lock<std::mutex> lock(_app->get_state_mutex(), std::try_to_lock);
    add("state_snapshot_available", lock.owns_lock());
    if (lock.owns_lock()) {
      const auto state = _app->get_state();
      const auto updates = _app->update_diagnostics();
      add("initialized_estimator", _app->initialized_estimator());
      add("initialized_output", _app->initialized());
      add("tracking_in_progress", _app->tracking_in_progress());
      add("state_stamp_s", state->_timestamp);
      add("state_age_s", message.header.stamp.toSec() - state->_timestamp);
      add("camera_imu_offset_s", state->_calib_dt_CAMtoIMU->value()(0));
      const auto bg = state->_imu->bias_g(), ba = state->_imu->bias_a(), velocity = state->_imu->vel();
      for (int i=0; i<3; ++i) {
        add("gyro_bias_" + std::to_string(i), bg(i));
        add("accel_bias_" + std::to_string(i), ba(i));
        add("velocity_global_" + std::to_string(i), velocity(i));
      }
      add("clones", state->_clones_IMU.size());
      add("slam_landmarks", state->_features_SLAM.size());
      add("zupt_accepted", updates.zupt_accepted);
      add("last_zupt_stamp_s", updates.last_zupt_stamp);
      add("moved_since_zupt", updates.moved_since_zupt);
      add("last_zupt", updates.last_zupt);
      add("msckf_gated_features_last", updates.msckf_gated_last);
      add("msckf_gated_features_total", updates.msckf_gated_total);
      add("last_msckf_stamp_s", updates.last_msckf_stamp);
      add("slam_gated_features_last", updates.slam_gated_last);
      add("slam_gated_features_total", updates.slam_gated_total);
      add("last_slam_stamp_s", updates.last_slam_stamp);
      add("camera_bundles", camera_sync_bundle_count_);
    } else {
      status.level = diagnostic_msgs::DiagnosticStatus::WARN;
      status.message = "estimator busy; state snapshot skipped without waiting";
    }
  }
  message.status.push_back(std::move(status));
  pub_recording_diagnostics_.publish(message);
}

void ROS1Visualizer::setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser) {

  // We need a valid parser
  assert(parser != nullptr);

  // Create imu subscriber (handle legacy ros param info)
  std::string topic_imu;
  _nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  sub_imu = _nh->subscribe(topic_imu, 1000, &ROS1Visualizer::callback_inertial, this, ros::TransportHints().tcpNoDelay());
  PRINT_INFO("subscribing to IMU: %s\n", topic_imu.c_str());

  // Logic for sync stereo subscriber
  // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
  if (expected_cameras_ == 2) {
    // Read in the topics
    std::string cam_topic0, cam_topic1;
    _nh->param<std::string>("topic_camera" + std::to_string(0), cam_topic0, "/cam" + std::to_string(0) + "/image_raw");
    _nh->param<std::string>("topic_camera" + std::to_string(1), cam_topic1, "/cam" + std::to_string(1) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", cam_topic0);
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", cam_topic1);
    // Create sync filter (they have unique pointers internally, so we have to use move logic here...)
    auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
        *_nh, cam_topic0, camera_subscriber_queue_size_, ros::TransportHints().tcpNoDelay());
    auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
        *_nh, cam_topic1, camera_subscriber_queue_size_, ros::TransportHints().tcpNoDelay());
    auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(10), *image_sub0, *image_sub1);
    sync->registerCallback(boost::bind(&ROS1Visualizer::callback_stereo, this, _1, _2, 0, 1));
    // Append to our vector of subscribers
    sync_cam.push_back(sync);
    sync_subs_cam.push_back(image_sub0);
    sync_subs_cam.push_back(image_sub1);
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic0.c_str());
    PRINT_INFO("subscribing to cam (stereo): %s\n", cam_topic1.c_str());
  } else {
    // Now we should add any non-stereo callbacks here
    for (size_t i = 0; i < expected_cameras_; i++) {
      // read in the topic
      std::string cam_topic;
      _nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
      parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
      // create subscriber
      subs_cam.push_back(_nh->subscribe<sensor_msgs::Image>(
          cam_topic, camera_subscriber_queue_size_,
          boost::bind(&ROS1Visualizer::callback_monocular, this, _1, i),
          ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay()));
      PRINT_INFO("subscribing to cam (mono): %s\n", cam_topic.c_str());
    }
  }
}

void ROS1Visualizer::visualize() {

  // Return if we have already visualized
  if (last_visualization_timestamp == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp = _app->get_state()->_timestamp;

  // Start timing
  // boost::posix_time::ptime rT0_1, rT0_2;
  // rT0_1 = boost::posix_time::microsec_clock::local_time();

  // publish current image (only if not multi-threaded)
  if (!async_pubs_)
    publish_images();

  // Return if we have not inited
  if (!_app->initialized())
    return;

  // Save the start time of this dataset
  if (!start_time_set) {
    rT1 = boost::posix_time::microsec_clock::local_time();
    start_time_set = true;
  }

  // publish state
  publish_state();

  // publish points
  publish_features();

  // Publish gt if we have it
  publish_groundtruth();

  // Publish keyframe information
  publish_loopclosure_information();

  // Save total state
  if (save_total_state) {
    ROSVisualizerHelper::sim_save_total_state_to_file(_app->get_state(), _sim, of_state_est, of_state_std, of_state_gt);
  }

  // Print how much time it took to publish / displaying things
  // rT0_2 = boost::posix_time::microsec_clock::local_time();
  // double time_total = (rT0_2 - rT0_1).total_microseconds() * 1e-6;
  // PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for visualization\n" RESET, time_total);
}

void ROS1Visualizer::visualize_odometry(double timestamp) {

  // Return if we have not inited
  if (!_app->initialized())
    return;

  const bool publish_odom = pub_odomimu.getNumSubscribers() != 0;
  const bool publish_tf = publish_global2imu_tf || publish_calibration_tf;
  if (!publish_odom && !publish_tf)
    return;

  // Get fast propagate state at the desired timestamp
  std::shared_ptr<State> state = _app->get_state();
  if (!std::isfinite(timestamp) || timestamp <= 0.0 || timestamp <= state->_timestamp)
    return;
  Eigen::Matrix<double, 13, 1> state_plus = Eigen::Matrix<double, 13, 1>::Zero();
  Eigen::Matrix<double, 12, 12> cov_plus = Eigen::Matrix<double, 12, 12>::Zero();
  if (!_app->get_propagator()->fast_state_propagate(state, timestamp, state_plus, cov_plus))
    return;

  const double quaternion_norm = state_plus.head<4>().norm();
  if (!state_plus.allFinite() || !cov_plus.allFinite() ||
      !std::isfinite(quaternion_norm) || quaternion_norm <= 1e-12) {
    _app->get_propagator()->invalidate_cache();
    invalid_fast_odom_count_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t now_ns = steady_now_ns();
    if (invalid_fast_odom_last_report_ns_ == 0 ||
        now_ns - invalid_fast_odom_last_report_ns_ >= 2000000000ULL) {
      invalid_fast_odom_last_report_ns_ = now_ns;
      ROS_ERROR_STREAM("Dropping invalid OpenVINS fast odometry: requested_stamp=" << timestamp
                       << " state_stamp=" << state->_timestamp
                       << " state_finite=" << state_plus.allFinite()
                       << " covariance_finite=" << cov_plus.allFinite()
                       << " quaternion_norm=" << quaternion_norm);
    }
    return;
  }

  //  // Get the simulated groundtruth so we can evaulate the error in respect to it
  //  // NOTE: we get the true time in the IMU clock frame
  //  if (_sim != nullptr) {
  //    Eigen::Matrix<double, 17, 1> state_gt;
  //    if (_sim->get_state(timestamp, state_gt)) {
  //      // Difference between positions
  //      double dx = state_plus(4, 0) - state_gt(5, 0);
  //      double dy = state_plus(5, 0) - state_gt(6, 0);
  //      double dz = state_plus(6, 0) - state_gt(7, 0);
  //      double err_pos = std::sqrt(dx * dx + dy * dy + dz * dz);
  //      // Quaternion error
  //      Eigen::Matrix<double, 4, 1> quat_gt, quat_st, quat_diff;
  //      quat_gt << state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0);
  //      quat_st << state_plus(0, 0), state_plus(1, 0), state_plus(2, 0), state_plus(3, 0);
  //      quat_diff = quat_multiply(quat_st, Inv(quat_gt));
  //      double err_ori = (180 / M_PI) * 2 * quat_diff.block(0, 0, 3, 1).norm();
  //      // Calculate NEES values
  //      Eigen::Vector3d quat_diff_vec = quat_diff.block(0, 0, 3, 1);
  //      Eigen::Vector3d cov_vec = cov_plus.block(0, 0, 3, 3).inverse() * 2 * quat_diff.block(0, 0, 3, 1);
  //      double ori_nees = 2 * quat_diff_vec.dot(cov_vec);
  //      Eigen::Vector3d errpos = state_plus.block(4, 0, 3, 1) - state_gt.block(5, 0, 3, 1);
  //      double pos_nees = errpos.transpose() * cov_plus.block(3, 3, 3, 3).inverse() * errpos;
  //      PRINT_INFO(REDPURPLE "error to gt => %.3f, %.3f (deg,m) | nees => %.1f, %.1f (ori,pos) \n" RESET, err_ori, err_pos, ori_nees,
  //                 pos_nees);
  //    }
  //  }

  // Publish our odometry message if requested
  if (publish_odom) {

    nav_msgs::Odometry odomIinM;
    odomIinM.header.stamp = ros::Time(timestamp);
    odomIinM.header.frame_id = "global";

    // The POSE component (orientation and position)
    odomIinM.pose.pose.orientation.x = state_plus(0);
    odomIinM.pose.pose.orientation.y = state_plus(1);
    odomIinM.pose.pose.orientation.z = state_plus(2);
    odomIinM.pose.pose.orientation.w = state_plus(3);
    odomIinM.pose.pose.position.x = state_plus(4);
    odomIinM.pose.pose.position.y = state_plus(5);
    odomIinM.pose.pose.position.z = state_plus(6);

    // The TWIST component (angular and linear velocities)
    odomIinM.child_frame_id = "imu";
    odomIinM.twist.twist.linear.x = state_plus(7);   // vel in local frame
    odomIinM.twist.twist.linear.y = state_plus(8);   // vel in local frame
    odomIinM.twist.twist.linear.z = state_plus(9);   // vel in local frame
    odomIinM.twist.twist.angular.x = state_plus(10); // we do not estimate this...
    odomIinM.twist.twist.angular.y = state_plus(11); // we do not estimate this...
    odomIinM.twist.twist.angular.z = state_plus(12); // we do not estimate this...

    // OpenVINS orders the first six covariance states as orientation then
    // position; ROS orders them as position then orientation. Reorder directly
    // instead of constructing and multiplying two dense 12x12 permutation
    // matrices for every IMU sample.
    static constexpr int kPoseCovarianceSource[6] = {3, 4, 5, 0, 1, 2};
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.pose.covariance[6 * r + c] =
            cov_plus(kPoseCovarianceSource[r], kPoseCovarianceSource[c]);
      }
    }
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        odomIinM.twist.covariance[6 * r + c] = cov_plus(r + 6, c + 6);
      }
    }
    pub_odomimu.publish(odomIinM);
  }

  // Publish each fast odometry transform once at its measurement time. The
  // simulation /clock can repeat while distinct IMU callbacks are delivered;
  // stamping each transform with ros::Time::now() floods TF_REPEATED_DATA.
  if (!publish_tf || timestamp <= last_odometry_tf_timestamp)
    return;
  last_odometry_tf_timestamp = timestamp;

  if (publish_global2imu_tf) {
    // Publish our transform on TF
    // NOTE: since we use JPL we have an implicit conversion to Hamilton when we publish
    // NOTE: a rotation from GtoI in JPL has the same xyzw as a ItoG Hamilton rotation
    auto odom_pose = std::make_shared<ov_type::PoseJPL>();
    odom_pose->set_value(state_plus.block(0, 0, 7, 1));
    tf::StampedTransform trans = ROSVisualizerHelper::get_stamped_transform_from_pose(odom_pose, false);
    trans.stamp_ = ros::Time(timestamp);
    trans.frame_id_ = "global";
    trans.child_frame_id_ = "imu";
    mTfBr->sendTransform(trans);
  }

  // Loop through each camera calibration and publish it
  if (publish_calibration_tf) {
    for (const auto &calib : state->_calib_IMUtoCAM) {
      tf::StampedTransform trans_calib = ROSVisualizerHelper::get_stamped_transform_from_pose(calib.second, true);
      trans_calib.stamp_ = ros::Time(timestamp);
      trans_calib.frame_id_ = "imu";
      trans_calib.child_frame_id_ = "cam" + std::to_string(calib.first);
      mTfBr->sendTransform(trans_calib);
    }
  }
}

void ROS1Visualizer::visualize_final() {
  // Drain callbacks in the ROS entry point before closing worker admission.
  // These joins must precede the state lock needed by the workers themselves.
  stop_workers();
  _app->wait_for_initialization();
  std::lock_guard<std::mutex> state_lock(_app->get_state_mutex());

  // Final time offset value
  if (_app->get_state()->_options.do_calib_camera_timeoffset) {
    PRINT_INFO(REDPURPLE "camera-imu timeoffset = %.5f\n\n" RESET, _app->get_state()->_calib_dt_CAMtoIMU->value()(0));
  }

  // Final camera intrinsics
  if (_app->get_state()->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = _app->get_state()->_cam_intrinsics.at(i);
      PRINT_INFO(REDPURPLE "cam%d intrinsics:\n" RESET, (int)i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n" RESET, calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));
      PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,%.5f\n\n" RESET, calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Final camera extrinsics
  if (_app->get_state()->_options.do_calib_camera_pose) {
    for (int i = 0; i < _app->get_state()->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = _app->get_state()->_calib_IMUtoCAM.at(i);
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = quat_2_Rot(calib->quat()).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * calib->pos();
      PRINT_INFO(REDPURPLE "T_C%dtoI:\n" RESET, i);
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(0, 0), T_CtoI(0, 1), T_CtoI(0, 2), T_CtoI(0, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(1, 0), T_CtoI(1, 1), T_CtoI(1, 2), T_CtoI(1, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f,\n" RESET, T_CtoI(2, 0), T_CtoI(2, 1), T_CtoI(2, 2), T_CtoI(2, 3));
      PRINT_INFO(REDPURPLE "%.3f,%.3f,%.3f,%.3f\n\n" RESET, T_CtoI(3, 0), T_CtoI(3, 1), T_CtoI(3, 2), T_CtoI(3, 3));
    }
  }

  // IMU intrinsics
  if (_app->get_state()->_options.do_calib_imu_intrinsics) {
    Eigen::Matrix3d Dw = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_dw->value());
    Eigen::Matrix3d Da = State::Dm(_app->get_state()->_options.imu_model, _app->get_state()->_calib_imu_da->value());
    Eigen::Matrix3d Tw = Dw.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d Ta = Da.colPivHouseholderQr().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d R_IMUtoACC = _app->get_state()->_calib_imu_ACCtoIMU->Rot().transpose();
    Eigen::Matrix3d R_IMUtoGYRO = _app->get_state()->_calib_imu_GYROtoIMU->Rot().transpose();
    PRINT_INFO(REDPURPLE "Tw:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(0, 0), Tw(0, 1), Tw(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Tw(1, 0), Tw(1, 1), Tw(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Tw(2, 0), Tw(2, 1), Tw(2, 2));
    PRINT_INFO(REDPURPLE "Ta:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(0, 0), Ta(0, 1), Ta(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, Ta(1, 0), Ta(1, 1), Ta(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, Ta(2, 0), Ta(2, 1), Ta(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoACC:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(0, 0), R_IMUtoACC(0, 1), R_IMUtoACC(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoACC(1, 0), R_IMUtoACC(1, 1), R_IMUtoACC(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoACC(2, 0), R_IMUtoACC(2, 1), R_IMUtoACC(2, 2));
    PRINT_INFO(REDPURPLE "R_IMUtoGYRO:\n" RESET);
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(0, 0), R_IMUtoGYRO(0, 1), R_IMUtoGYRO(0, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f,\n" RESET, R_IMUtoGYRO(1, 0), R_IMUtoGYRO(1, 1), R_IMUtoGYRO(1, 2));
    PRINT_INFO(REDPURPLE "%.5f,%.5f,%.5f\n\n" RESET, R_IMUtoGYRO(2, 0), R_IMUtoGYRO(2, 1), R_IMUtoGYRO(2, 2));
  }

  // IMU intrinsics gravity sensitivity
  if (_app->get_state()->_options.do_calib_imu_g_sensitivity) {
    Eigen::Matrix3d Tg = State::Tg(_app->get_state()->_calib_imu_tg->value());
    PRINT_INFO(REDPURPLE "Tg:\n" RESET);
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(0, 0), Tg(0, 1), Tg(0, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f,\n" RESET, Tg(1, 0), Tg(1, 1), Tg(1, 2));
    PRINT_INFO(REDPURPLE "%.6f,%.6f,%.6f\n\n" RESET, Tg(2, 0), Tg(2, 1), Tg(2, 2));
  }

  // Publish RMSE if we have it
  if (!gt_states.empty()) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
  }

  // Publish RMSE and NEES if doing simulation
  if (_sim != nullptr) {
    PRINT_INFO(REDPURPLE "RMSE: %.3f (deg) orientation\n" RESET, std::sqrt(summed_mse_ori / summed_number));
    PRINT_INFO(REDPURPLE "RMSE: %.3f (m) position\n\n" RESET, std::sqrt(summed_mse_pos / summed_number));
    PRINT_INFO(REDPURPLE "NEES: %.3f (deg) orientation\n" RESET, summed_nees_ori / summed_number);
    PRINT_INFO(REDPURPLE "NEES: %.3f (m) position\n\n" RESET, summed_nees_pos / summed_number);
  }

  // Print the total time
  rT2 = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO(REDPURPLE "TIME: %.3f seconds\n\n" RESET, (rT2 - rT1).total_microseconds() * 1e-6);
}

void ROS1Visualizer::callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {
  if (stopping_.load())
    return;
  const uint64_t receipt_ns = steady_now_ns();
  imu_callback_count_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t previous_receipt_ns = imu_last_receipt_ns_.exchange(receipt_ns, std::memory_order_relaxed);
  if (previous_receipt_ns != 0 && receipt_ns >= previous_receipt_ns) {
    update_timing_max(imu_max_receipt_gap_ns_, receipt_ns - previous_receipt_ns);
  }

  // convert into correct format
  ov_core::ImuData message;
  message.timestamp = msg->header.stamp.toSec();
  const PhaseSnapshot state_lock_start = phase_snapshot();
  std::unique_lock<std::mutex> state_lock(_app->get_state_mutex());
  const PhaseSnapshot feed_start =
      record_timing(imu_state_lock_wait_timing_, "imu_state_lock_wait",
                    state_lock_start, message.timestamp);
  // Keep the dynamic initializer on the raw united stream. A vibration
  // filter can otherwise change the excitation/noise statistics used to
  // recover the initial attitude and biases. Once OpenVINS reports that it
  // is initialized, seed the filter and enable it for normal propagation.
  const bool apply_imu_pre_filter = imu_pre_filter_enable_ && _app->initialized();
  if (apply_imu_pre_filter && !imu_pre_filter_active_) {
    imu_pre_filter_.reset();
    imu_pre_filter_active_ = true;
  } else if (!apply_imu_pre_filter) {
    imu_pre_filter_active_ = false;
  }
  if (apply_imu_pre_filter) {
    double wm[3] = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
    double am[3] = {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z};
    double filtered_wm[3], filtered_am[3];
    imu_pre_filter_.filter(wm, am, filtered_wm, filtered_am);
    message.wm << filtered_wm[0], filtered_wm[1], filtered_wm[2];
    message.am << filtered_am[0], filtered_am[1], filtered_am[2];
  } else {
    message.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    message.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  }

  // send it to our VIO system
  _app->feed_measurement_imu(message);
  const PhaseSnapshot visualize_start =
      record_timing(imu_feed_timing_, "feed_measurement_imu", feed_start,
                    message.timestamp);
  visualize_odometry(message.timestamp);
  record_timing(imu_visualize_timing_, "visualize_odometry", visualize_start, message.timestamp);
  // Camera workers acquire queue -> state. Release state before this callback
  // can start or join a worker that needs the camera queue.
  state_lock.unlock();

  {
    std::lock_guard<std::mutex> lock(camera_queue_mtx);
    latest_imu_timestamp_ = message.timestamp;
    ++camera_queue_generation_;
  }
  camera_queue_cv_.notify_one();
  if (!async_subs_)
    process_camera_queue();
}

void ROS1Visualizer::process_camera_queue() {
  // Queue ownership covers selection/removal only. The estimator and all ROS
  // publications run after it is released, so camera ingress never waits for
  // an entire visual update or a catch-up loop.
  while (!stopping_.load()) {
    double latest_imu;
    {
      std::lock_guard<std::mutex> lock(camera_queue_mtx);
      if (camera_queue.empty())
        return;
      latest_imu = latest_imu_timestamp_;
    }
    double timestamp_imu_inC;
    {
      std::lock_guard<std::mutex> state_lock(_app->get_state_mutex());
      timestamp_imu_inC = latest_imu - _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
    }
    CameraInput update_input;
    bool bundled = false;
    double bundle_span = 0;
    {
      std::lock_guard<std::mutex> lock(camera_queue_mtx);
      trim_camera_queue(ros::Time::now().toSec());
      if (camera_queue.empty() || camera_queue.front().timestamp >= timestamp_imu_inC)
        return;
      std::map<int, bool> unique_ids;
      for (const auto &camera : camera_queue)
        unique_ids[camera.sensor_ids.front()] = true;
      const size_t streams = expected_cameras_ == 2 ? 1 : expected_cameras_;
      if (unique_ids.size() < streams)
        return;
      const auto &first = camera_queue.front();
      update_input = first;
      std::vector<size_t> consumed{0};
      if (camera_sync_tolerance_s_ > 0 && expected_cameras_ > 2 && first.sensor_ids.size() == 1) {
        const double group_end = first.timestamp + camera_sync_tolerance_s_;
        std::map<int, size_t> members;
        std::map<int, double> latest;
        for (size_t i = 0; i < camera_queue.size(); ++i) {
          const auto &candidate = camera_queue[i];
          if (candidate.sensor_ids.size() != 1)
            continue;
          const int id = candidate.sensor_ids.front();
          if (id < 0 || static_cast<size_t>(id) >= expected_cameras_)
            continue;
          latest[id] = candidate.timestamp;
          if (candidate.timestamp <= group_end)
            members.emplace(id, i);
        }
        if (members.size() == expected_cameras_) {
          double max_stamp = first.timestamp, sum = 0;
          update_input.sensor_ids.clear();
          update_input.images.clear();
          consumed.clear();
          for (const auto &member : members) {
            const auto &camera = camera_queue[member.second];
            max_stamp = std::max(max_stamp, camera.timestamp);
            sum += camera.timestamp;
            update_input.sensor_ids.push_back(member.first);
            update_input.images.push_back(camera.images.front());
            consumed.push_back(member.second);
          }
          if (max_stamp >= timestamp_imu_inC)
            return;
          update_input.timestamp = sum / static_cast<double>(members.size());
          bundle_span = max_stamp - first.timestamp;
          bundled = true;
        } else if (camera_bundle_wait_for_missing(expected_cameras_, members, latest, group_end)) {
          return;
        }
      }
      std::sort(consumed.begin(), consumed.end());
      for (auto i = consumed.rbegin(); i != consumed.rend(); ++i)
        camera_queue.erase(camera_queue.begin() + *i);
    }
    // Ingress retains the ROS buffers. Convert only selected, fresh frames,
    // outside the estimator lock; discarded frames incur no pixel copy.
    ov_core::CameraData update_message;
    update_message.timestamp = update_input.timestamp;
    update_message.sensor_ids = update_input.sensor_ids;
    try {
      for (size_t i = 0; i < update_input.images.size(); ++i) {
        const auto image = cv_bridge::toCvCopy(update_input.images[i], sensor_msgs::image_encodings::MONO8);
        update_message.images.push_back(image->image);
        const int id = update_input.sensor_ids[i];
        if (!camera_masks_.empty()) {
          update_message.masks.push_back(camera_masks_.at(id));
        } else if (id >= 0 && static_cast<size_t>(id) < zero_camera_masks_.size() &&
                   zero_camera_masks_.at(id).size() == image->image.size()) {
          update_message.masks.push_back(zero_camera_masks_.at(id));
        } else {
          update_message.masks.push_back(cv::Mat::zeros(image->image.size(), CV_8UC1));
        }
      }
    } catch (const cv_bridge::Exception &error) {
      PRINT_ERROR("cv_bridge exception: %s\n", error.what());
      continue;
    }

    const PhaseSnapshot lock_start = phase_snapshot();
    std::unique_lock<std::mutex> state_lock(_app->get_state_mutex());
    const PhaseSnapshot feed_start =
        record_timing(camera_state_lock_wait_timing_, "camera_state_lock_wait",
                      lock_start, update_message.timestamp);
    // Waiting for the state lock can invalidate a previously fresh selection.
    if (camera_input_expired(update_input.timestamp, ros::Time::now().toSec())) {
      std::lock_guard<std::mutex> queue_lock(camera_queue_mtx);
      camera_dropped_through_ = std::max(camera_dropped_through_, update_input.timestamp + camera_sync_tolerance_s_);
      camera_queue_dropped_.fetch_add(update_input.images.size(), std::memory_order_relaxed);
      camera_queue_expired_.fetch_add(update_input.images.size(), std::memory_order_relaxed);
      continue;
    }
    const bool need_depth = it_pub_loop_img_depth.getNumSubscribers() != 0 ||
                            it_pub_loop_img_depth_color.getNumSubscribers() != 0;
    _app->set_active_track_output(!demand_driven_loop_output_ || need_depth ||
        pub_loop_pose.getNumSubscribers() != 0 || pub_loop_point.getNumSubscribers() != 0,
        !demand_driven_loop_output_ || need_depth);
    if (camera_tracking_outside_state_lock_)
      _app->feed_measurement_camera(update_message, state_lock);
    else
      _app->feed_measurement_camera(update_message);
    const PhaseSnapshot vis_start =
        record_timing(camera_feed_timing_, "feed_measurement_camera", feed_start,
                      update_message.timestamp);
    visualize();
    record_timing(camera_visualize_timing_, "visualize_camera_update", vis_start, update_message.timestamp);
    const double age = ros::Time::now().toSec() - update_message.timestamp;
    if (age > 0 && std::isfinite(age))
      update_timing_max(camera_update_age_max_ns_, static_cast<uint64_t>(age * 1e9));
    if (bundled && ++camera_sync_bundle_count_ % 100 == 1) {
      char record[256];
      const int n = std::snprintf(record, sizeof(record), "Camera sync bundles: %zu (latest span %.3f s)\n",
                                  camera_sync_bundle_count_, bundle_span);
      if (n > 0 && static_cast<size_t>(n) < sizeof(record))
        _app->enqueue_info_diagnostic(record, static_cast<size_t>(n));
    }
  }
}

bool ROS1Visualizer::camera_input_expired(double timestamp, double now) const {
  return camera_queue_max_age_s_ > 0 && std::isfinite(now) && now > 0 &&
         now - (timestamp + camera_sync_tolerance_s_) > camera_queue_max_age_s_;
}

void ROS1Visualizer::trim_camera_queue(double now) {
  const size_t expired = trim_camera_backlog(camera_queue, 0, 0, camera_sync_tolerance_s_,
      camera_dropped_through_, now, camera_queue_max_age_s_);
  const size_t overflow = trim_camera_backlog(camera_queue, static_cast<size_t>(camera_queue_max_messages_),
      camera_queue_max_span_s_, camera_sync_tolerance_s_, camera_dropped_through_);
  camera_queue_expired_.fetch_add(expired, std::memory_order_relaxed);
  camera_queue_dropped_.fetch_add(expired + overflow, std::memory_order_relaxed);
}

void ROS1Visualizer::enqueue_camera(CameraInput message, const char *phase) {
  const double timestamp = message.timestamp;
  const auto start = phase_snapshot();
  {
    std::lock_guard<std::mutex> lock(camera_queue_mtx);
    record_timing(camera_queue_lock_timing_, phase, start, timestamp);
    if (timestamp <= camera_dropped_through_) {
      camera_queue_dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const auto where = std::upper_bound(camera_queue.begin(), camera_queue.end(), message);
    camera_queue.insert(where, std::move(message));
    trim_camera_queue(ros::Time::now().toSec());
    update_timing_max(camera_queue_peak_, camera_queue.size());
    ++camera_queue_generation_;
  }
  camera_queue_cv_.notify_one();
}

void ROS1Visualizer::callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0) {
  if (stopping_.load())
    return;

  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = camera_min_interval_s_;
  {
    std::lock_guard<std::mutex> timestamp_lock(camera_last_timestamp_mtx);
    if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta)
      return;
    camera_last_timestamp[cam_id0] = timestamp;
  }
  CameraInput message;
  message.timestamp = timestamp;
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(msg0);
  enqueue_camera(std::move(message), "camera_queue_lock_monocular");
}

void ROS1Visualizer::callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0,
                                     int cam_id1) {
  if (stopping_.load())
    return;

  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = camera_min_interval_s_;
  {
    std::lock_guard<std::mutex> timestamp_lock(camera_last_timestamp_mtx);
    if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta) {
      return;
    }
    camera_last_timestamp[cam_id0] = timestamp;
  }

  CameraInput message;
  message.timestamp = timestamp;
  message.sensor_ids = {cam_id0, cam_id1};
  message.images = {msg0, msg1};
  enqueue_camera(std::move(message), "camera_queue_lock_stereo");
}

void ROS1Visualizer::publish_state() {

  const bool publish_pose = pub_poseimu.getNumSubscribers() != 0;
  const bool publish_path = pub_pathimu.getNumSubscribers() != 0;
  const bool retain_path = publish_path || path_history_without_subscribers_;
  if (!publish_pose && !retain_path)
    return;

  // Get the current state
  std::shared_ptr<State> state = _app->get_state();

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = state->_timestamp + t_ItoC;

  // Create pose of IMU (note we use the bag time)
  geometry_msgs::PoseWithCovarianceStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_imu;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.pose.orientation.x = state->_imu->quat()(0);
  poseIinM.pose.pose.orientation.y = state->_imu->quat()(1);
  poseIinM.pose.pose.orientation.z = state->_imu->quat()(2);
  poseIinM.pose.pose.orientation.w = state->_imu->quat()(3);
  poseIinM.pose.pose.position.x = state->_imu->pos()(0);
  poseIinM.pose.pose.position.y = state->_imu->pos()(1);
  poseIinM.pose.pose.position.z = state->_imu->pos()(2);

  if (publish_pose) {
    // Finally set the covariance in the message (in the order position then orientation as per ros convention)
    std::vector<std::shared_ptr<Type>> statevars;
    statevars.push_back(state->_imu->pose()->p());
    statevars.push_back(state->_imu->pose()->q());
    Eigen::Matrix<double, 6, 6> covariance_posori = StateHelper::get_marginal_covariance(_app->get_state(), statevars);
    for (int r = 0; r < 6; r++) {
      for (int c = 0; c < 6; c++) {
        poseIinM.pose.covariance[6 * r + c] = covariance_posori(r, c);
      }
    }
    pub_poseimu.publish(poseIinM);
  }

  //=========================================================
  //=========================================================

  if (retain_path) {
    // Append to our pose vector
    geometry_msgs::PoseStamped posetemp;
    posetemp.header = poseIinM.header;
    posetemp.pose = poseIinM.pose.pose;
    poses_imu.push_back(posetemp);
  }

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  if (publish_path) {
    nav_msgs::Path arrIMU;
    arrIMU.header.stamp = ros::Time::now();
    arrIMU.header.seq = poses_seq_imu;
    arrIMU.header.frame_id = "global";
    for (size_t i = 0; i < poses_imu.size(); i += std::floor((double)poses_imu.size() / 16384.0) + 1)
      arrIMU.poses.push_back(poses_imu.at(i));
    pub_pathimu.publish(arrIMU);
  }

  // Move them forward in time
  poses_seq_imu++;
}

void ROS1Visualizer::publish_images() {

  // Return if we have already visualized
  if (_app->get_state() == nullptr)
    return;
  if (last_visualization_timestamp_image == _app->get_state()->_timestamp && _app->initialized())
    return;
  last_visualization_timestamp_image = _app->get_state()->_timestamp;

  // Check if we have subscribers
  if (it_pub_tracks.getNumSubscribers() == 0)
    return;

  // Get our image of history tracks
  cv::Mat img_history = _app->get_historical_viz_image();
  if (img_history.empty())
    return;

  // Create our message
  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = "cam0";
  sensor_msgs::ImagePtr exl_msg = cv_bridge::CvImage(header, "bgr8", img_history).toImageMsg();

  // Publish
  it_pub_tracks.publish(exl_msg);
}

void ROS1Visualizer::publish_features() {

  // Get our good MSCKF features
  if (pub_points_msckf.getNumSubscribers() != 0) {
    std::vector<Eigen::Vector3d> feats_msckf = _app->get_good_features_MSCKF();
    sensor_msgs::PointCloud2 cloud = ROSVisualizerHelper::get_ros_pointcloud(feats_msckf);
    pub_points_msckf.publish(cloud);
  }

  // Get our good SLAM features
  if (pub_points_slam.getNumSubscribers() != 0) {
    std::vector<Eigen::Vector3d> feats_slam = _app->get_features_SLAM();
    sensor_msgs::PointCloud2 cloud_SLAM = ROSVisualizerHelper::get_ros_pointcloud(feats_slam);
    pub_points_slam.publish(cloud_SLAM);
  }

  // Get our good ARUCO features
  if (pub_points_aruco.getNumSubscribers() != 0) {
    std::vector<Eigen::Vector3d> feats_aruco = _app->get_features_ARUCO();
    sensor_msgs::PointCloud2 cloud_ARUCO = ROSVisualizerHelper::get_ros_pointcloud(feats_aruco);
    pub_points_aruco.publish(cloud_ARUCO);
  }

  // Skip the rest of we are not doing simulation
  if (_sim == nullptr)
    return;

  // Get our good SIMULATION features
  if (pub_points_sim.getNumSubscribers() != 0) {
    std::vector<Eigen::Vector3d> feats_sim = _sim->get_map_vec();
    sensor_msgs::PointCloud2 cloud_SIM = ROSVisualizerHelper::get_ros_pointcloud(feats_sim);
    pub_points_sim.publish(cloud_SIM);
  }
}

void ROS1Visualizer::publish_groundtruth() {

  if (_sim == nullptr && gt_states.empty())
    return;

  // Our groundtruth state
  Eigen::Matrix<double, 17, 1> state_gt;

  // We want to publish in the IMU clock frame
  // The timestamp in the state will be the last camera time
  double t_ItoC = _app->get_state()->_calib_dt_CAMtoIMU->value()(0);
  double timestamp_inI = _app->get_state()->_timestamp + t_ItoC;

  // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
  if (_sim == nullptr && (gt_states.empty() || !DatasetReader::get_gt_state(timestamp_inI, state_gt, gt_states))) {
    return;
  }

  // Get the simulated groundtruth
  // NOTE: we get the true time in the IMU clock frame
  if (_sim != nullptr) {
    timestamp_inI = _app->get_state()->_timestamp + _sim->get_true_parameters().calib_camimu_dt;
    if (!_sim->get_state(timestamp_inI, state_gt))
      return;
  }

  // Get the GT and system state state
  Eigen::Matrix<double, 16, 1> state_ekf = _app->get_state()->_imu->value();

  // Create pose of IMU
  geometry_msgs::PoseStamped poseIinM;
  poseIinM.header.stamp = ros::Time(timestamp_inI);
  poseIinM.header.seq = poses_seq_gt;
  poseIinM.header.frame_id = "global";
  poseIinM.pose.orientation.x = state_gt(1, 0);
  poseIinM.pose.orientation.y = state_gt(2, 0);
  poseIinM.pose.orientation.z = state_gt(3, 0);
  poseIinM.pose.orientation.w = state_gt(4, 0);
  poseIinM.pose.position.x = state_gt(5, 0);
  poseIinM.pose.position.y = state_gt(6, 0);
  poseIinM.pose.position.z = state_gt(7, 0);
  pub_posegt.publish(poseIinM);

  // Append to our pose vector
  poses_gt.push_back(poseIinM);

  // Create our path (imu)
  // NOTE: We downsample the number of poses as needed to prevent rviz crashes
  // NOTE: https://github.com/ros-visualization/rviz/issues/1107
  nav_msgs::Path arrIMU;
  arrIMU.header.stamp = ros::Time::now();
  arrIMU.header.seq = poses_seq_gt;
  arrIMU.header.frame_id = "global";
  for (size_t i = 0; i < poses_gt.size(); i += std::floor((double)poses_gt.size() / 16384.0) + 1) {
    arrIMU.poses.push_back(poses_gt.at(i));
  }
  pub_pathgt.publish(arrIMU);

  // Move them forward in time
  poses_seq_gt++;

  // Publish our transform on TF
  tf::StampedTransform trans;
  trans.stamp_ = ros::Time::now();
  trans.frame_id_ = "global";
  trans.child_frame_id_ = "truth";
  tf::Quaternion quat(state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0));
  trans.setRotation(quat);
  tf::Vector3 orig(state_gt(5, 0), state_gt(6, 0), state_gt(7, 0));
  trans.setOrigin(orig);
  if (publish_global2imu_tf) {
    mTfBr->sendTransform(trans);
  }

  //==========================================================================
  //==========================================================================

  // Difference between positions
  double dx = state_ekf(4, 0) - state_gt(5, 0);
  double dy = state_ekf(5, 0) - state_gt(6, 0);
  double dz = state_ekf(6, 0) - state_gt(7, 0);
  double err_pos = std::sqrt(dx * dx + dy * dy + dz * dz);

  // Quaternion error
  Eigen::Matrix<double, 4, 1> quat_gt, quat_st, quat_diff;
  quat_gt << state_gt(1, 0), state_gt(2, 0), state_gt(3, 0), state_gt(4, 0);
  quat_st << state_ekf(0, 0), state_ekf(1, 0), state_ekf(2, 0), state_ekf(3, 0);
  quat_diff = quat_multiply(quat_st, Inv(quat_gt));
  double err_ori = (180 / M_PI) * 2 * quat_diff.block(0, 0, 3, 1).norm();

  //==========================================================================
  //==========================================================================

  // Get covariance of pose
  std::vector<std::shared_ptr<Type>> statevars;
  statevars.push_back(_app->get_state()->_imu->q());
  statevars.push_back(_app->get_state()->_imu->p());
  Eigen::Matrix<double, 6, 6> covariance = StateHelper::get_marginal_covariance(_app->get_state(), statevars);

  // Calculate NEES values
  // NOTE: need to manually multiply things out to make static asserts work
  // NOTE: https://github.com/rpng/open_vins/pull/226
  // NOTE: https://github.com/rpng/open_vins/issues/236
  // NOTE: https://gitlab.com/libeigen/eigen/-/issues/1664
  Eigen::Vector3d quat_diff_vec = quat_diff.block(0, 0, 3, 1);
  Eigen::Vector3d cov_vec = covariance.block(0, 0, 3, 3).inverse() * 2 * quat_diff.block(0, 0, 3, 1);
  double ori_nees = 2 * quat_diff_vec.dot(cov_vec);
  Eigen::Vector3d errpos = state_ekf.block(4, 0, 3, 1) - state_gt.block(5, 0, 3, 1);
  double pos_nees = errpos.transpose() * covariance.block(3, 3, 3, 3).inverse() * errpos;

  //==========================================================================
  //==========================================================================

  // Update our average variables
  if (!std::isnan(ori_nees) && !std::isnan(pos_nees)) {
    summed_mse_ori += err_ori * err_ori;
    summed_mse_pos += err_pos * err_pos;
    summed_nees_ori += ori_nees;
    summed_nees_pos += pos_nees;
    summed_number++;
  }

  // Nice display for the user
  PRINT_INFO(REDPURPLE "error to gt => %.3f, %.3f (deg,m) | rmse => %.3f, %.3f (deg,m) | called %d times\n" RESET, err_ori, err_pos,
             std::sqrt(summed_mse_ori / summed_number), std::sqrt(summed_mse_pos / summed_number), (int)summed_number);
  PRINT_INFO(REDPURPLE "nees => %.1f, %.1f (ori,pos) | avg nees = %.1f, %.1f (ori,pos)\n" RESET, ori_nees, pos_nees,
             summed_nees_ori / summed_number, summed_nees_pos / summed_number);

  //==========================================================================
  //==========================================================================
}

// Publish the IMU->camera extrinsic and intrinsics of every camera so the
// pose graph can start. This deliberately runs ahead of the active-track
// gates in publish_loopclosure_information(): on a stationary start every
// camera frame takes the ZUPT branch of VioManager::track_image_and_update
// and returns before do_feature_propagate_update(), so active_tracks_time
// stays -1 until the vehicle first moves. Behind those gates, loop_fusion
// blocks forever in its startup calibration wait exactly when it should be
// ready for the first takeoff. The estimator state timestamp is unusable as
// a rate limiter here for the same reason, so this uses wall time.
void ROS1Visualizer::publish_loop_calibration() {
  const auto state = _app->get_state();
  if (state->_calib_IMUtoCAM.empty() || state->_cam_intrinsics.empty())
    return;

  bool subscribed = false;
  for (size_t c = 0; c < pub_loop_intrinsics.size(); ++c) {
    subscribed = subscribed || pub_loop_extrinsic.at(c).getNumSubscribers() != 0 ||
                 pub_loop_intrinsics.at(c).getNumSubscribers() != 0;
  }
  if (!subscribed)
    return;

  const double now = ros::Time::now().toSec();
  if (last_loop_calibration_time >= 0.0 && now - last_loop_calibration_time < 0.2)
    return;
  last_loop_calibration_time = now;

  std_msgs::Header header;
  header.stamp = ros::Time(now);

  const size_t num_cams = std::min(std::min(_app->get_params().camera_intrinsics.size(),
                                            state->_calib_IMUtoCAM.size()),
                                   state->_cam_intrinsics.size());
  for (size_t c = 0; c < num_cams && c < pub_loop_intrinsics.size(); ++c) {

    // need to flip the transform to the IMU frame
    Eigen::Vector4d q_ItoC = state->_calib_IMUtoCAM.at(c)->quat();
    Eigen::Vector3d p_CinI = -state->_calib_IMUtoCAM.at(c)->Rot().transpose() *
                             state->_calib_IMUtoCAM.at(c)->pos();
    nav_msgs::Odometry odometry_calib;
    odometry_calib.header = header;
    odometry_calib.header.frame_id = "imu";
    odometry_calib.pose.pose.position.x = p_CinI(0);
    odometry_calib.pose.pose.position.y = p_CinI(1);
    odometry_calib.pose.pose.position.z = p_CinI(2);
    odometry_calib.pose.pose.orientation.x = q_ItoC(0);
    odometry_calib.pose.pose.orientation.y = q_ItoC(1);
    odometry_calib.pose.pose.orientation.z = q_ItoC(2);
    odometry_calib.pose.pose.orientation.w = q_ItoC(3);
    pub_loop_extrinsic.at(c).publish(odometry_calib);

    // PUBLISH CAMERA INTRINSICS
    bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(_app->get_params().camera_intrinsics.at(c)) != nullptr);
    sensor_msgs::CameraInfo cameraparams;
    cameraparams.header = header;
    cameraparams.header.frame_id = "cam" + std::to_string(c);
    cameraparams.width = static_cast<uint32_t>(_app->get_params().camera_intrinsics.at(c)->w());
    cameraparams.height = static_cast<uint32_t>(_app->get_params().camera_intrinsics.at(c)->h());
    cameraparams.distortion_model = is_fisheye ? "equidistant" : "plumb_bob";
    Eigen::VectorXd cparams = state->_cam_intrinsics.at(c)->value();
    cameraparams.D = {cparams(4), cparams(5), cparams(6), cparams(7)};
    cameraparams.K = {cparams(0), 0, cparams(2), 0, cparams(1), cparams(3), 0, 0, 1};
    pub_loop_intrinsics.at(c).publish(cameraparams);
  }
}

void ROS1Visualizer::publish_loopclosure_information() {

  // Calibration must never wait for an active-track sample. On a stationary
  // start every camera frame takes the ZUPT branch in
  // VioManager::track_image_and_update and returns before
  // do_feature_propagate_update(), so active_tracks_time stays -1 until the
  // vehicle first moves. Publishing calibration here is what lets
  // loop_fusion finish its startup wait while the vehicle is still on the
  // ground; ROS2Visualizer publishes it ahead of the track gates for the
  // same reason.
  publish_loop_calibration();

  const bool publish_pose = pub_loop_pose.getNumSubscribers() != 0;
  const bool publish_points = pub_loop_point.getNumSubscribers() != 0;
  const bool need_depth = it_pub_loop_img_depth.getNumSubscribers() != 0 ||
                          it_pub_loop_img_depth_color.getNumSubscribers() != 0;
  const bool need_active_tracks = publish_pose || publish_points || need_depth;
  if (!need_active_tracks)
    return;

  // Gate on the timestamp before copying the potentially large feature maps.
  // The estimator runs much faster than the 5 Hz loop-fusion output.
  const double active_tracks_time1 = _app->get_active_tracks_time();
  if (active_tracks_time1 == -1)
    return;
  if (_app->get_state()->_clones_IMU.find(active_tracks_time1) == _app->get_state()->_clones_IMU.end())
    return;
  // Keep the ROS1 split-simulation output rate aligned with the ROS2 visualizer.
  // The estimator updates on IMU messages, but loop fusion only needs one
  // synchronized image/feature/pose sample every 0.2 seconds.
  if (last_loopclosure_publish_time >= 0.0 &&
      active_tracks_time1 >= last_loopclosure_publish_time &&
      active_tracks_time1 - last_loopclosure_publish_time < 0.2)
    return;

  double active_tracks_time2 = active_tracks_time1;
  cv::Mat active_cam0_image;
  if (need_depth) {
    _app->get_active_image(active_tracks_time2, active_cam0_image);
    if (active_tracks_time1 != active_tracks_time2)
      return;
  }
  last_loopclosure_publish_time = active_tracks_time1;

  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  if (publish_points || need_depth) {
    double copied_tracks_time = -1;
    _app->get_active_tracks(copied_tracks_time, active_tracks_posinG, active_tracks_uvd);
    if (copied_tracks_time != active_tracks_time1)
      return;
  }

  // Default header
  std_msgs::Header header;
  header.stamp = ros::Time(active_tracks_time1);

  //======================================================
  // PUBLISH HISTORICAL POSE ESTIMATE
  // Per-camera calibration is published by publish_loop_calibration() before
  // the active-track gates above, so it is intentionally not gated here.
  if (publish_pose) {
    Eigen::Vector4d quat = _app->get_state()->_clones_IMU.at(active_tracks_time1)->quat();
    Eigen::Vector3d pos = _app->get_state()->_clones_IMU.at(active_tracks_time1)->pos();
    nav_msgs::Odometry odometry_pose;
    odometry_pose.header = header;
    odometry_pose.header.frame_id = "global";
    odometry_pose.pose.pose.position.x = pos(0);
    odometry_pose.pose.pose.position.y = pos(1);
    odometry_pose.pose.pose.position.z = pos(2);
    odometry_pose.pose.pose.orientation.x = quat(0);
    odometry_pose.pose.pose.orientation.y = quat(1);
    odometry_pose.pose.pose.orientation.z = quat(2);
    odometry_pose.pose.pose.orientation.w = quat(3);
    pub_loop_pose.publish(odometry_pose);
  }

  //======================================================
  // PUBLISH FEATURE TRACKS IN THE GLOBAL FRAME OF REFERENCE
  if (publish_points) {

    // Construct the message
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    point_cloud.header.frame_id = "global";
    for (const auto &feattimes : active_tracks_posinG) {

      // Get this feature information
      size_t featid = feattimes.first;
      const Eigen::Vector3d pFinG = feattimes.second;
      if (!pFinG.allFinite())
        continue;

      // cam0 pixels stay real observations so the cam0 keyframe keeps using the
      // measured uv. Landmarks only visible to the other cameras are published
      // with a (-1,-1) placeholder; loop_fusion projects those with each
      // camera's own model and extrinsic instead of fabricating (0,0) matches.
      double uv_x = -1.0;
      double uv_y = -1.0;
      const auto observation = active_tracks_uvd.find(featid);
      if (observation != active_tracks_uvd.end()) {
        const Eigen::Vector3d &uvd = observation->second;
        if (uvd.allFinite() && uvd(2) >= 0.1) {
          uv_x = uvd(0);
          uv_y = uvd(1);
        }
      }

      // Push back 3d point
      geometry_msgs::Point32 p;
      p.x = pFinG(0);
      p.y = pFinG(1);
      p.z = pFinG(2);
      point_cloud.points.push_back(p);

      // Push back the uv_norm, uv_raw, and feature id
      // NOTE: we don't use the normalized coordinates to save time here
      // NOTE: they will have to be re-normalized in the loop closure code
      sensor_msgs::ChannelFloat32 p_2d;
      p_2d.values.push_back(0);
      p_2d.values.push_back(0);
      p_2d.values.push_back(uv_x);
      p_2d.values.push_back(uv_y);
      p_2d.values.push_back(featid);
      point_cloud.channels.push_back(p_2d);
    }
    pub_loop_point.publish(point_cloud);
  }

  //======================================================
  // Depth images of sparse points and its colorized version
  if (need_depth) {

    // Create the images we will populate with the depths
    std::pair<int, int> wh_pair = {active_cam0_image.cols, active_cam0_image.rows};
    cv::Mat depthmap = cv::Mat::zeros(wh_pair.second, wh_pair.first, CV_16UC1);
    cv::Mat depthmap_viz = active_cam0_image;

    // Loop through all points and append
    for (const auto &feattimes : active_tracks_uvd) {

      // Get this feature information
      size_t featid = feattimes.first;
      Eigen::Vector3d uvd = active_tracks_uvd.at(featid);

      // Skip invalid points
      double dw = 4;
      if (uvd(0) < dw || uvd(0) > wh_pair.first - dw || uvd(1) < dw || uvd(1) > wh_pair.second - dw) {
        continue;
      }

      // Append the depth
      // NOTE: scaled by 1000 to fit the 16U
      // NOTE: access order is y,x (stupid opencv convention stuff)
      depthmap.at<uint16_t>((int)uvd(1), (int)uvd(0)) = (uint16_t)(1000 * uvd(2));

      // Taken from LSD-SLAM codebase segment into 0-4 meter segments:
      // https://github.com/tum-vision/lsd_slam/blob/d1e6f0e1a027889985d2e6b4c0fe7a90b0c75067/lsd_slam_core/src/util/globalFuncs.cpp#L87-L96
      float id = 1.0f / (float)uvd(2);
      float r = (0.0f - id) * 255 / 1.0f;
      if (r < 0)
        r = -r;
      float g = (1.0f - id) * 255 / 1.0f;
      if (g < 0)
        g = -g;
      float b = (2.0f - id) * 255 / 1.0f;
      if (b < 0)
        b = -b;
      uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
      uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
      uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);
      cv::Scalar color(255 - rc, 255 - gc, 255 - bc);

      // Small square around the point (note the above bound check needs to take into account this width)
      cv::Point p0(uvd(0) - dw, uvd(1) - dw);
      cv::Point p1(uvd(0) + dw, uvd(1) + dw);
      cv::rectangle(depthmap_viz, p0, p1, color, -1);
    }

    // Create our messages
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_msg1 = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_16UC1, depthmap).toImageMsg();
    it_pub_loop_img_depth.publish(exl_msg1);
    header.stamp = ros::Time::now();
    header.frame_id = "cam0";
    sensor_msgs::ImagePtr exl_msg2 = cv_bridge::CvImage(header, "bgr8", depthmap_viz).toImageMsg();
    it_pub_loop_img_depth_color.publish(exl_msg2);
  }
}

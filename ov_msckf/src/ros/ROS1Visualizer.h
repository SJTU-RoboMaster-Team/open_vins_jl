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

#ifndef OV_MSCKF_ROS1VISUALIZER_H
#define OV_MSCKF_ROS1VISUALIZER_H

#include <diagnostic_msgs/DiagnosticArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>

#include "utils/ImuPreFilter.h"

namespace ov_core {
class YamlParser;
struct CameraData;
} // namespace ov_core

namespace ov_msckf {

class VioManager;
class Simulator;

/**
 * @brief Helper class that will publish results onto the ROS framework.
 *
 * Also save to file the current total state and covariance along with the groundtruth if we are simulating.
 * We visualize the following things:
 * - State of the system on TF, pose message, and path
 * - Image of our tracker
 * - Our different features (SLAM, MSCKF, ARUCO)
 * - Groundtruth trajectory if we have it
 */
class ROS1Visualizer {

public:
  /**
   * @brief Default constructor
   * @param nh ROS node handler
   * @param app Core estimator manager
   * @param sim Simulator if we are simulating
   */
  ROS1Visualizer(std::shared_ptr<ros::NodeHandle> nh, std::shared_ptr<VioManager> app, std::shared_ptr<Simulator> sim = nullptr);
  ~ROS1Visualizer();

  /**
   * @brief Will setup ROS subscribers and callbacks
   * @param parser Configuration file parser
   */
  void setup_subscribers(std::shared_ptr<ov_core::YamlParser> parser);

  /**
   * @brief Will visualize the system if we have new things
   */
  void visualize();

  /**
   * @brief Will publish our odometry message for the current timestep.
   * This will take the current state estimate and get the propagated pose to the desired time.
   * This can be used to get pose estimates on systems which require high frequency pose estimates.
   */
  void visualize_odometry(double timestamp);

  /**
   * @brief After the run has ended, print results
   */
  void visualize_final();

  /// Callback for inertial information
  void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg);

  /// Callback for monocular cameras information
  void callback_monocular(const sensor_msgs::ImageConstPtr &msg0, int cam_id0);

  /// Callback for synchronized stereo camera information
  void callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0, int cam_id1);

protected:
  /// Publish the current state
  void publish_state();

  /// Publish the active tracking image
  void publish_images();

  /// Publish current features
  void publish_features();

  /// Publish groundtruth (if we have it)
  void publish_groundtruth();

  /// Publish loop-closure information of current pose and active track information
  void publish_loopclosure_information();

  /// Wall-clock performance diagnostics; these only observe existing callbacks.
  struct TimingStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> max_ns{0};
  };
  struct PhaseSnapshot {
    uint64_t steady_ns{0};
    uint64_t cpu_ns{0};
    long minor_faults{0};
    long major_faults{0};
    long voluntary_switches{0};
    long involuntary_switches{0};
  };
  static uint64_t steady_now_ns();
  static void update_timing_max(std::atomic<uint64_t> &target, uint64_t value);
  static PhaseSnapshot phase_snapshot();
  PhaseSnapshot record_timing(TimingStats &stats, const char *phase,
                              const PhaseSnapshot &start, double sensor_stamp);
  void report_timing_diagnostics(const ros::WallTimerEvent &event);
  void publish_recording_diagnostics(const ros::WallTimerEvent &event);

  /// Publish per-camera loop calibration independently of the active tracks.
  /// See the definition for why it must not sit behind the track gates.
  void publish_loop_calibration();

  /// Caller must quiesce external ROS callbacks before destroying this object.
  /// Joins never run while holding estimator state or camera queue locks.
  void stop_workers();
  void process_camera_queue();
  struct CameraInput {
    double timestamp = 0;
    std::vector<int> sensor_ids;
    std::vector<sensor_msgs::ImageConstPtr> images;
    bool operator<(const CameraInput &other) const {
      return timestamp == other.timestamp ? sensor_ids.front() < other.sensor_ids.front() : timestamp < other.timestamp;
    }
  };
  void enqueue_camera(CameraInput message, const char *phase);
  void trim_camera_queue(double now);
  bool camera_input_expired(double timestamp, double now) const;

  /// Global node handler
  std::shared_ptr<ros::NodeHandle> _nh;

  /// Core application of the filter system
  std::shared_ptr<VioManager> _app;

  /// Simulator (is nullptr if we are not sim'ing)
  std::shared_ptr<Simulator> _sim;

  // Our publishers
  image_transport::Publisher it_pub_tracks, it_pub_loop_img_depth, it_pub_loop_img_depth_color;
  ros::Publisher pub_poseimu, pub_odomimu, pub_pathimu;
  ros::Publisher pub_recording_diagnostics_;
  ros::Publisher pub_stereo_points_;
  ros::Publisher pub_points_msckf, pub_points_slam, pub_points_aruco, pub_points_sim;
  ros::Publisher pub_loop_pose, pub_loop_point;
  // One calibration publisher per camera. Camera 0 keeps the historical
  // `loop_extrinsic`/`loop_intrinsics` topic names; camera N uses `_camN`.
  std::vector<ros::Publisher> pub_loop_extrinsic, pub_loop_intrinsics;
  double last_loopclosure_publish_time = -1.0;
  double last_loop_calibration_time = -1.0;
  std::shared_ptr<tf::TransformBroadcaster> mTfBr;

  // Our subscribers and camera synchronizers
  ros::Subscriber sub_imu;
  std::vector<ros::Subscriber> subs_cam;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  std::vector<std::shared_ptr<message_filters::Synchronizer<sync_pol>>> sync_cam;
  std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;

  // For path viz
  unsigned int poses_seq_imu = 0;
  std::vector<geometry_msgs::PoseStamped> poses_imu;

  // Groundtruth infomation
  ros::Publisher pub_pathgt, pub_posegt;
  double summed_mse_ori = 0.0;
  double summed_mse_pos = 0.0;
  double summed_nees_ori = 0.0;
  double summed_nees_pos = 0.0;
  size_t summed_number = 0;

  // Start and end timestamps
  bool start_time_set = false;
  boost::posix_time::ptime rT1, rT2;

  // Thread atomics
  bool async_subs_ = false;
  bool async_pubs_ = false;
  std::size_t expected_cameras_ = 1;
  double camera_min_interval_s_ = 0;
  std::map<size_t, cv::Mat> camera_masks_;
  // Immutable zero masks are shared when no configured mask is present.
  // TrackKLT only reads these masks, so allocating one per image is unnecessary.
  std::vector<cv::Mat> zero_camera_masks_;
  std::atomic<bool> stopping_{false};
  std::mutex worker_ownership_mtx_;
  std::thread image_thread_, update_thread_;
  bool workers_stopped_ = false;
  ros::WallTimer timing_diagnostics_timer_, recording_diagnostics_timer_;
  std::atomic<uint64_t> imu_callback_count_{0};
  std::atomic<uint64_t> imu_last_receipt_ns_{0};
  std::atomic<uint64_t> imu_max_receipt_gap_ns_{0};
  TimingStats imu_feed_timing_;
  TimingStats imu_visualize_timing_;
  TimingStats camera_feed_timing_;
  TimingStats camera_visualize_timing_;
  TimingStats camera_queue_lock_timing_;
  TimingStats imu_state_lock_wait_timing_;
  TimingStats camera_state_lock_wait_timing_;
  TimingStats image_state_lock_wait_timing_;
  std::atomic<uint64_t> invalid_fast_odom_count_{0};
  uint64_t invalid_fast_odom_last_report_ns_{0};

  // Optional T265 vibration pre-filter. Disabled by default; experiments
  // enable it through private ROS parameters without changing the estimator
  // YAML or the stock replay behavior.
  ov_core::ImuPreFilter imu_pre_filter_;
  bool imu_pre_filter_enable_ = false;
  bool imu_pre_filter_active_ = false;

  /// Queue up camera measurements sorted by time and trigger once we have
  /// exactly one IMU measurement with timestamp newer than the camera measurement
  /// This also handles out-of-order camera measurements, which is rare, but
  /// a nice feature to have for general robustness to bad camera drivers.
  std::deque<CameraInput> camera_queue;
  std::mutex camera_queue_mtx;
  std::condition_variable camera_queue_cv_;
  uint64_t camera_queue_generation_ = 0;
  double latest_imu_timestamp_ = -1;
  int camera_queue_max_messages_ = 16;
  int camera_subscriber_queue_size_ = 2;
  double camera_queue_max_span_s_ = 0.15;
  double camera_queue_max_age_s_ = 0.15;
  bool camera_tracking_outside_state_lock_ = true;
  bool demand_driven_loop_output_ = true;
  bool stereo_points_enabled_ = false;
  double camera_dropped_through_ = -1;
  std::atomic<uint64_t> camera_queue_dropped_{0};
  std::atomic<uint64_t> camera_queue_expired_{0};
  std::atomic<uint64_t> camera_queue_peak_{0};
  std::atomic<uint64_t> camera_update_age_max_ns_{0};
  // Optional tolerance for treating near-synchronous monocular cameras as one
  // estimator update. Zero preserves the stock one-message-per-update behavior.
  double camera_sync_tolerance_s_ = 0.0;
  size_t camera_sync_bundle_count_ = 0;

  // Last camera message timestamps we have received (mapped by cam id)
  std::map<int, double> camera_last_timestamp;
  std::mutex camera_last_timestamp_mtx;

  // Last timestamp we visualized at
  double last_visualization_timestamp = 0;
  double last_visualization_timestamp_image = 0;

  // TF shares the fast odometry sensor stamp; duplicate IMU samples must not
  // broadcast an already published transform again.
  double last_odometry_tf_timestamp = 0;

  // Our groundtruth states
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;

  // For path viz
  unsigned int poses_seq_gt = 0;
  std::vector<geometry_msgs::PoseStamped> poses_gt;
  bool publish_global2imu_tf = true;
  bool publish_calibration_tf = true;
  bool path_history_without_subscribers_ = true;

  // Files and if we should save total state
  bool save_total_state = false;
  std::ofstream of_state_est, of_state_std, of_state_gt;
};

} // namespace ov_msckf

#endif // OV_MSCKF_ROS1VISUALIZER_H

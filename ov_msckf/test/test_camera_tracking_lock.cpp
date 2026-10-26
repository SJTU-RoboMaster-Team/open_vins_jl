#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/VioManager.h"
#include "cam/CamRadtan.h"
#include "state/State.h"
#include "track/TrackBase.h"
#include "utils/sensor_data.h"

namespace {

const std::chrono::seconds kTrackingTimeout(3);
const std::chrono::seconds kTestTimeout(2);

ov_msckf::VioManagerOptions camera_test_options() {
  ov_msckf::VioManagerOptions options;
  options.state_options.num_cameras = 1;
  options.state_options.max_clone_size = 5;
  options.state_options.max_slam_features = 0;
  options.use_aruco = false;
  options.use_klt = true;
  options.use_stereo = false;
  options.histogram_method = ov_core::TrackBase::NONE;
  options.num_opencv_threads = 1;

  auto camera = std::make_shared<ov_core::CamRadtan>(64, 48);
  Eigen::Matrix<double, 8, 1> intrinsics;
  intrinsics << 40.0, 40.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0;
  camera->set_value(intrinsics);
  options.camera_intrinsics.emplace(0, camera);

  Eigen::Matrix<double, 7, 1> extrinsics;
  extrinsics << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  options.camera_extrinsics.emplace(0, extrinsics);
  options.init_options.num_cameras = 1;
  options.init_options.camera_intrinsics = options.camera_intrinsics;
  options.init_options.camera_extrinsics = options.camera_extrinsics;

  options.vec_dw << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
  options.vec_da << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
  options.vec_tg.setZero();
  options.q_GYROtoIMU << 0.0, 0.0, 0.0, 1.0;
  options.q_ACCtoIMU << 0.0, 0.0, 0.0, 1.0;
  return options;
}

class BlockingTracker final : public ov_core::TrackBase {
public:
  explicit BlockingTracker(
      std::unordered_map<std::size_t, std::shared_ptr<ov_core::CamBase>> cameras)
      : TrackBase(std::move(cameras), 0, 0, false, HistogramMethod::NONE) {}

  void feed_new_camera(const ov_core::CameraData &) override {
    ++feed_count_;
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    if (block_)
      cv_.wait_for(lock, kTrackingTimeout, [this] { return released_; });
    if (throw_)
      throw std::runtime_error("fake tracker failure");
  }

  bool wait_until_entered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, kTestTimeout, [this] { return entered_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

  void throw_after_entry() {
    block_ = false;
    throw_ = true;
  }
  int feed_count() const { return feed_count_.load(); }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool block_ = true;
  bool entered_ = false;
  bool released_ = false;
  bool throw_ = false;
  std::atomic<int> feed_count_{0};
};

class TestVioManager final : public ov_msckf::VioManager {
public:
  explicit TestVioManager(ov_msckf::VioManagerOptions &options)
      : VioManager(options) {}

  std::shared_ptr<BlockingTracker> install_blocking_tracker() {
    tracker_ = std::make_shared<BlockingTracker>(params.camera_intrinsics);
    trackFEATS = tracker_;
    return tracker_;
  }

private:
  std::shared_ptr<BlockingTracker> tracker_;
};

struct CameraFixture {
  ov_msckf::VioManagerOptions options = camera_test_options();
  TestVioManager manager{options};

  CameraFixture() {
    auto lock = std::unique_lock<std::mutex>(manager.get_state_mutex());
    for (int i = 0; i <= 1; ++i) {
      ov_core::ImuData imu;
      imu.timestamp = 0.99 + 0.01 * i;
      imu.wm.setZero();
      imu.am << 0.0, 0.0, 9.81;
      manager.feed_measurement_imu(imu);
    }

    Eigen::Matrix<double, 17, 1> initial = Eigen::Matrix<double, 17, 1>::Zero();
    initial(0) = 1.0;
    initial(4) = 1.0;  // JPL quaternion scalar component.
    manager.initialize_with_gt(initial);
  }

  ov_core::CameraData camera_message(double timestamp = 1.05) const {
    ov_core::CameraData message;
    message.timestamp = timestamp;
    message.sensor_ids = {0};
    message.images = {cv::Mat::zeros(48, 64, CV_8UC1)};
    message.masks = {cv::Mat::zeros(48, 64, CV_8UC1)};
    return message;
  }
};

TEST(CameraTrackingLock, ReleasesForFrontendAndUpdatesAfterIMUCanFeed) {
  CameraFixture fixture;
  auto tracker = fixture.manager.install_blocking_tracker();
  const auto message = fixture.camera_message();
  bool camera_lock_restored = false;
  std::exception_ptr camera_error;

  std::thread camera_thread([&] {
    std::unique_lock<std::mutex> lock(fixture.manager.get_state_mutex());
    try {
      fixture.manager.feed_measurement_camera(message, lock);
      camera_lock_restored = lock.owns_lock();
    } catch (...) {
      camera_error = std::current_exception();
    }
  });

  const bool tracker_entered = tracker->wait_until_entered();
  double timestamp_before_imu = -1.0;
  double timestamp_after_imu = -1.0;
  bool imu_thread_acquired = false;
  std::thread imu_thread;
  std::promise<void> imu_done_promise;
  std::future<void> imu_done;
  if (tracker_entered) {
    imu_done = imu_done_promise.get_future();
    imu_thread = std::thread([&] {
      std::unique_lock<std::mutex> lock(fixture.manager.get_state_mutex());
      imu_thread_acquired = true;
      timestamp_before_imu = fixture.manager.get_state()->_timestamp;
      for (int i = 1; i <= 10; ++i) {
        ov_core::ImuData imu;
        imu.timestamp = 1.0 + 0.01 * i;
        imu.wm.setZero();
        imu.am << 0.0, 0.0, 9.81;
        fixture.manager.feed_measurement_imu(imu);
      }
      timestamp_after_imu = fixture.manager.get_state()->_timestamp;
      imu_done_promise.set_value();
    });
  }

  const bool imu_finished_while_tracker_blocked =
      tracker_entered && imu_done.wait_for(kTestTimeout) == std::future_status::ready;
  if (!imu_finished_while_tracker_blocked)
    tracker->release();
  if (imu_thread.joinable())
    imu_thread.join();
  tracker->release();
  camera_thread.join();

  EXPECT_TRUE(tracker_entered);
  EXPECT_TRUE(imu_finished_while_tracker_blocked);
  EXPECT_TRUE(imu_thread_acquired);
  EXPECT_DOUBLE_EQ(1.0, timestamp_before_imu);
  EXPECT_DOUBLE_EQ(timestamp_before_imu, timestamp_after_imu);
  EXPECT_TRUE(camera_lock_restored);
  EXPECT_EQ(nullptr, camera_error);

  std::lock_guard<std::mutex> lock(fixture.manager.get_state_mutex());
  const auto state = fixture.manager.get_state();
  EXPECT_DOUBLE_EQ(message.timestamp, state->_timestamp);
  EXPECT_TRUE(state->_imu->quat().allFinite());
  EXPECT_TRUE(state->_imu->pos().allFinite());
  EXPECT_TRUE(state->_imu->vel().allFinite());
}

TEST(CameraTrackingLock, RestoresCallerLockWhenTrackerThrows) {
  CameraFixture fixture;
  auto tracker = fixture.manager.install_blocking_tracker();
  tracker->throw_after_entry();
  auto lock = std::unique_lock<std::mutex>(fixture.manager.get_state_mutex());

  EXPECT_THROW(fixture.manager.feed_measurement_camera(fixture.camera_message(), lock),
               std::runtime_error);
  EXPECT_TRUE(lock.owns_lock());
  EXPECT_EQ(1, tracker->feed_count());
}

TEST(CameraTrackingLock, RejectsWrongAndUnownedLocks) {
  CameraFixture fixture;
  auto tracker = fixture.manager.install_blocking_tracker();
  const auto message = fixture.camera_message();

  std::mutex unrelated_mutex;
  std::unique_lock<std::mutex> wrong_lock(unrelated_mutex);
  EXPECT_THROW(fixture.manager.feed_measurement_camera(message, wrong_lock),
               std::invalid_argument);

  std::unique_lock<std::mutex> unowned_lock(fixture.manager.get_state_mutex(),
                                            std::defer_lock);
  EXPECT_THROW(fixture.manager.feed_measurement_camera(message, unowned_lock),
               std::invalid_argument);
  EXPECT_EQ(0, tracker->feed_count());
}

TEST(CameraTrackingLock, LegacyFeedRemainsSerializedByCallerLock) {
  CameraFixture fixture;
  auto tracker = fixture.manager.install_blocking_tracker();
  const auto message = fixture.camera_message();
  std::thread legacy_thread([&] {
    std::unique_lock<std::mutex> lock(fixture.manager.get_state_mutex());
    fixture.manager.feed_measurement_camera(message);
  });

  const bool tracker_entered = tracker->wait_until_entered();
  bool competing_thread_acquired = false;
  if (tracker_entered) {
    std::thread competing_thread([&] {
      std::unique_lock<std::mutex> lock(fixture.manager.get_state_mutex(),
                                        std::try_to_lock);
      competing_thread_acquired = lock.owns_lock();
    });
    competing_thread.join();
  }

  tracker->release();
  legacy_thread.join();
  EXPECT_TRUE(tracker_entered);
  EXPECT_FALSE(competing_thread_acquired);
}

}  // namespace

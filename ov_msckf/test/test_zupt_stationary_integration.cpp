#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "feat/FeatureDatabase.h"
#include "feat/Feature.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "track/TrackBase.h"
#include "update/UpdaterOptions.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/NoiseManager.h"
#include "utils/sensor_data.h"

namespace {

using ov_core::FeatureDatabase;
using ov_core::ImuData;
using ov_msckf::NoiseManager;
using ov_msckf::Propagator;
using ov_msckf::State;
using ov_msckf::StateOptions;
using ov_msckf::UpdaterOptions;
using ov_msckf::UpdaterZeroVelocity;

ov_msckf::VioManagerOptions manager_options() {
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
  options.init_options.init_window_time = 2.0;
  options.vec_dw << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
  options.vec_da << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;
  options.vec_tg.setZero();
  options.q_GYROtoIMU << 0.0, 0.0, 0.0, 1.0;
  options.q_ACCtoIMU << 0.0, 0.0, 0.0, 1.0;
  options.try_zupt = true;
  options.zupt_require_visual_stationary = true;
  options.zupt_max_disparity = 0.5;
  return options;
}

class TestVioManager final : public ov_msckf::VioManager {
public:
  explicit TestVioManager(ov_msckf::VioManagerOptions &options)
      : VioManager(options) {}

  using VioManager::state;
  using VioManager::trackFEATS;
  using VioManager::updaterZUPT;

  void finish_static_initialization(double timestamp) {
    state->_timestamp = timestamp;
    is_initialized_vio = true;
  }
};

struct ZuptFixture {
  StateOptions state_options;
  NoiseManager noises;
  UpdaterOptions updater_options;
  std::shared_ptr<State> state;
  std::shared_ptr<FeatureDatabase> database;
  std::shared_ptr<Propagator> propagator;
  std::unique_ptr<UpdaterZeroVelocity> updater;

  ZuptFixture(bool require_visual_stationary, double max_disparity,
              double speed, double accel_residual, double disparity) {
    state = std::make_shared<State>(state_options);
    state->_timestamp = 1.0;
    state->_imu->v()->set_value(Eigen::Vector3d(speed, 0.0, 0.0));

    database = std::make_shared<FeatureDatabase>();
    for (std::size_t feature = 0; feature < 25; ++feature) {
      const float u = static_cast<float>(feature * 4);
      const float v = static_cast<float>(feature * 2);
      database->update_feature(feature, 1.0, 0, u, v, u, v);
      database->update_feature(feature, 1.1, 0, u + static_cast<float>(disparity),
                               v, u + static_cast<float>(disparity), v);
      database->get_feature(feature)->to_delete = false;
    }

    propagator = std::make_shared<Propagator>(noises, 9.81);
    for (int index = 0; index <= 10; ++index) {
      ImuData sample;
      sample.timestamp = 1.0 + 0.01 * index;
      sample.wm.setZero();
      sample.am << 0.0, 0.0, 9.81 + accel_residual;
      updater_imu_.push_back(sample);
    }

    updater.reset(new UpdaterZeroVelocity(
        updater_options, noises, database, propagator, 9.81, 0.1, 1.0,
        max_disparity, require_visual_stationary));
    for (const ImuData& sample : updater_imu_) updater->feed_imu(sample);
  }

 private:
  std::vector<ImuData> updater_imu_;
};

TEST(UpdaterZeroVelocityIntegration, LegacyVisualOverrideStillAcceptsPassingImu) {
  ZuptFixture fixture(false, 0.0, 0.02, 0.0, 1.0);

  EXPECT_TRUE(fixture.updater->try_update(fixture.state, 1.1));
  EXPECT_DOUBLE_EQ(fixture.state->_timestamp, 1.1);
}

TEST(UpdaterZeroVelocityIntegration, OptInRejectsImuPassWhenTrackedFeaturesMove) {
  ZuptFixture fixture(true, 0.5, 0.02, 0.0, 1.0);

  EXPECT_FALSE(fixture.updater->try_update(fixture.state, 1.1));
  EXPECT_DOUBLE_EQ(fixture.state->_timestamp, 1.0);
}

TEST(UpdaterZeroVelocityIntegration, OptInAcceptsVisualAndInertialStillness) {
  ZuptFixture fixture(true, 0.5, 0.02, 0.0, 0.1);

  EXPECT_TRUE(fixture.updater->try_update(fixture.state, 1.1));
  EXPECT_DOUBLE_EQ(fixture.state->_timestamp, 1.1);
}

TEST(UpdaterZeroVelocityIntegration, OptInRejectsVisualStillnessAboveSpeedLimit) {
  ZuptFixture fixture(true, 0.5, 0.2, 0.0, 0.1);

  EXPECT_FALSE(fixture.updater->try_update(fixture.state, 1.1));
  EXPECT_DOUBLE_EQ(fixture.state->_timestamp, 1.0);
}

TEST(UpdaterZeroVelocityIntegration, OptInRejectsVisualStillnessWithImuResidual) {
  ZuptFixture fixture(true, 0.5, 0.02, 1.0, 0.1);

  EXPECT_FALSE(fixture.updater->try_update(fixture.state, 1.1));
  EXPECT_DOUBLE_EQ(fixture.state->_timestamp, 1.0);
}

TEST(VioManagerZuptIntegration, FirstUpdateHasPreInitializationImuHistory) {
  auto options = manager_options();
  TestVioManager manager(options);

  // Startup IMU samples arrive before the inertial initializer accepts its
  // static window. The first ZUPT then consumes that same window.
  for (int index = 0; index <= 20; ++index) {
    ImuData sample;
    sample.timestamp = 0.90 + 0.01 * index;
    sample.wm.setZero();
    sample.am << 0.0, 0.0, 9.81;
    manager.feed_measurement_imu(sample);
  }

  for (std::size_t feature = 0; feature < 25; ++feature) {
    const float u = static_cast<float>(feature * 4);
    const float v = static_cast<float>(feature * 2);
    manager.trackFEATS->get_feature_database()->update_feature(
        feature, 1.0, 0, u, v, u, v);
    manager.trackFEATS->get_feature_database()->update_feature(
        feature, 1.1, 0, u + 0.1F, v, u + 0.1F, v);
    manager.trackFEATS->get_feature_database()->get_feature(feature)->to_delete = false;
  }

  manager.finish_static_initialization(1.0);

  ASSERT_NE(manager.updaterZUPT, nullptr);
  EXPECT_TRUE(manager.updaterZUPT->try_update(manager.state, 1.1));
  EXPECT_DOUBLE_EQ(manager.state->_timestamp, 1.1);
}

}  // namespace

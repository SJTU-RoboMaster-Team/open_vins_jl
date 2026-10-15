#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "feat/FeatureDatabase.h"
#include "feat/Feature.h"
#include "state/Propagator.h"
#include "state/State.h"
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

}  // namespace

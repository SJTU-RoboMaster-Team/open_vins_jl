#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterRange.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

namespace {

using ov_core::RangeData;
using ov_msckf::RangeAidOptions;
using ov_msckf::RangeUpdateStatus;
using ov_msckf::State;
using ov_msckf::StateOptions;
using ov_msckf::UpdaterRange;

constexpr double kPi = 3.14159265358979323846;

RangeAidOptions options() {
  RangeAidOptions result;
  result.enabled = true;
  result.R_range_to_imu = Eigen::Matrix3d::Identity();
  result.R_range_to_imu(1, 1) = -1.0;
  result.R_range_to_imu(2, 2) = -1.0;
  result.p_range_in_imu = Eigen::Vector3d::Zero();
  result.measurement_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
  result.min_range_m = 0.2;
  result.max_range_m = 5.0;
  result.noise_m = 0.05;
  result.max_age_s = 0.10;
  result.max_tilt_deg = 25.0;
  result.chi2_multiplier = 1.0;
  return result;
}

std::shared_ptr<State> make_state(double z = 1.0) {
  StateOptions state_options;
  auto state = std::make_shared<State>(state_options);
  Eigen::Matrix<double, 16, 1> imu_value = Eigen::Matrix<double, 16, 1>::Zero();
  imu_value(3) = 1.0;
  imu_value(6) = z;
  state->_imu->set_value(imu_value);
  Eigen::MatrixXd covariance =
      1e-6 * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
  covariance(5, 5) = 1.0;
  ov_msckf::StateHelper::set_initial_covariance(
      state, covariance,
      {state->_imu});
  state->_timestamp = 1.0;
  return state;
}

RangeData sample(double timestamp, double range) {
  RangeData result;
  result.timestamp = timestamp;
  result.range = range;
  result.min_range = 0.2;
  result.max_range = 5.0;
  return result;
}

void initialize_plane(UpdaterRange &updater, const std::shared_ptr<State> &state,
                      double plane_height = 0.0) {
  for (int index = 0; index < 5; ++index) {
    const double timestamp = 1.0 + 0.05 * index;
    state->_timestamp = timestamp;
    const double range = state->_imu->pos().z() - plane_height;
    const auto status = updater.try_update(state, sample(timestamp, range));
    if (index < 4) {
      EXPECT_EQ(status, RangeUpdateStatus::kPlanePending);
    } else {
      EXPECT_EQ(status, RangeUpdateStatus::kPlaneInitialized);
    }
  }
  EXPECT_TRUE(updater.diagnostics().plane_initialized);
  EXPECT_NEAR(updater.diagnostics().ground_plane_height_m, plane_height, 1e-9);
}

TEST(UpdaterRangeGeometry, PredictsVerticalRangeAndPlaneHeight) {
  const Eigen::Vector3d position(0.0, 0.0, 1.0);
  const Eigen::Vector3d ray(0.0, 0.0, -1.0);
  EXPECT_DOUBLE_EQ(ov_msckf::estimate_ground_plane_height(position, ray, 2.0),
                   -1.0);
  const auto prediction =
      ov_msckf::predict_planar_range(position, ray, -1.0,
                                     25.0 * kPi / 180.0);
  ASSERT_TRUE(prediction.valid);
  EXPECT_DOUBLE_EQ(prediction.range_m, 2.0);
  EXPECT_DOUBLE_EQ(prediction.ray_vertical, -1.0);
  EXPECT_DOUBLE_EQ(prediction.tilt_rad, 0.0);
}

TEST(UpdaterRangeGeometry, PredictedRangeGrowsWithTilt) {
  const Eigen::Vector3d position(0.0, 0.0, 1.0);
  const double tilt = 20.0 * kPi / 180.0;
  const Eigen::Vector3d ray(std::sin(tilt), 0.0, -std::cos(tilt));
  const auto prediction =
      ov_msckf::predict_planar_range(position, ray, -1.0,
                                     25.0 * kPi / 180.0);
  ASSERT_TRUE(prediction.valid);
  EXPECT_NEAR(prediction.range_m, 2.0 / std::cos(tilt), 1e-12);
  EXPECT_NEAR(prediction.tilt_rad, tilt, 1e-12);
}

TEST(UpdaterRangeGeometry, RejectsUpwardAndExcessiveTilt) {
  const Eigen::Vector3d position(0.0, 0.0, 1.0);
  EXPECT_FALSE(
      ov_msckf::predict_planar_range(position, Eigen::Vector3d(0, 0, 1), -1.0,
                                     25.0 * kPi / 180.0)
          .valid);
  const double tilt = 30.0 * kPi / 180.0;
  EXPECT_FALSE(
      ov_msckf::predict_planar_range(
          position, Eigen::Vector3d(std::sin(tilt), 0.0, -std::cos(tilt)),
          -1.0,
          25.0 * kPi / 180.0)
          .valid);
}

TEST(UpdaterRangeIntegration, InitializesThenCorrectsOnlyVerticalPosition) {
  auto state = make_state();
  UpdaterRange updater(options());
  initialize_plane(updater, state);

  const Eigen::Vector2d horizontal_before = state->_imu->pos().head<2>();
  Eigen::Matrix<double, 16, 1> drifted_imu = state->_imu->value();
  drifted_imu(6) = 1.5;
  state->_imu->set_value(drifted_imu);
  state->_timestamp = 1.25;
  const auto status = updater.try_update(state, sample(1.25, 1.0));
  EXPECT_EQ(status, RangeUpdateStatus::kAccepted);
  EXPECT_NEAR(state->_imu->pos().z(), 1.00125, 5e-4);
  EXPECT_DOUBLE_EQ(state->_imu->pos().x(), horizontal_before.x());
  EXPECT_DOUBLE_EQ(state->_imu->pos().y(), horizontal_before.y());
  EXPECT_EQ(updater.diagnostics().accepted, 2u);
}

TEST(UpdaterRangeIntegration, RejectsStaleOutOfRangeAndExcessiveTilt) {
  auto state = make_state();
  UpdaterRange updater(options());
  initialize_plane(updater, state);

  state->_timestamp = 1.50;
  EXPECT_EQ(updater.try_update(state, sample(1.20, 2.0)),
            RangeUpdateStatus::kRejected);
  EXPECT_EQ(updater.try_update(state, sample(1.50, 0.1)),
            RangeUpdateStatus::kRejected);

  const double tilt = 30.0 * kPi / 180.0;
  Eigen::Matrix3d R_GtoI = Eigen::Matrix3d::Identity();
  R_GtoI << std::cos(tilt), 0.0, -std::sin(tilt), 0.0, 1.0, 0.0,
      std::sin(tilt), 0.0, std::cos(tilt);
  state->_imu->q()->set_value(ov_core::rot_2_quat(R_GtoI));
  state->_timestamp = 1.55;
  EXPECT_EQ(updater.try_update(state, sample(1.55, 2.0)),
            RangeUpdateStatus::kRejected);
  EXPECT_EQ(updater.diagnostics().rejected, 3u);
}

} // namespace

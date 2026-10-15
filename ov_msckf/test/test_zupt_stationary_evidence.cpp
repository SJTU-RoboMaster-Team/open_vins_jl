#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "update/zupt_stationary_evidence.h"

namespace {

using ov_msckf::StationaryEvidence;
using ov_msckf::evaluate_stationary_evidence;

StationaryEvidence evaluate(double disparity, int feature_count, double max_disparity,
                            double chi2, double chi2_limit, double speed,
                            double max_speed) {
  return evaluate_stationary_evidence(disparity, feature_count, max_disparity,
                                      chi2, chi2_limit, speed, max_speed);
}

TEST(ZuptStationaryEvidence, RejectsPassingImuWhenVisualMotionIsTooLarge) {
  const StationaryEvidence result = evaluate(1.0, 100, 0.5, 0.5, 1.0, 0.02, 0.05);
  EXPECT_FALSE(result.visual_stationary);
  EXPECT_TRUE(result.inertial_stationary);
  EXPECT_FALSE(result.accepted);
}

TEST(ZuptStationaryEvidence, RejectsVisualStillnessWhenInertialEvidenceFails) {
  const StationaryEvidence result = evaluate(0.1, 100, 0.5, 1.1, 1.0, 0.06, 0.05);
  EXPECT_TRUE(result.visual_stationary);
  EXPECT_FALSE(result.inertial_stationary);
  EXPECT_FALSE(result.accepted);
}

TEST(ZuptStationaryEvidence, AcceptsEvidenceWhenBothSensorsAreStationary) {
  const StationaryEvidence result = evaluate(0.1, 100, 0.5, 0.5, 1.0, 0.02, 0.05);
  EXPECT_TRUE(result.visual_stationary);
  EXPECT_TRUE(result.inertial_stationary);
  EXPECT_TRUE(result.accepted);
}

TEST(ZuptStationaryEvidence, UsesStrictVisualThresholdAndRequiresMoreThanTwentyFeatures) {
  EXPECT_FALSE(evaluate(0.5, 100, 0.5, 0.0, 1.0, 0.0, 0.05).visual_stationary);
  EXPECT_FALSE(evaluate(0.1, 20, 0.5, 0.0, 1.0, 0.0, 0.05).visual_stationary);
  EXPECT_TRUE(evaluate(0.1, 21, 0.5, 0.0, 1.0, 0.0, 0.05).visual_stationary);
}

TEST(ZuptStationaryEvidence, AcceptsInertialThresholdEqualityAndZeroMotion) {
  const StationaryEvidence boundary = evaluate(0.0, 21, 0.5, 1.0, 1.0, 0.05, 0.05);
  EXPECT_TRUE(boundary.inertial_stationary);
  EXPECT_TRUE(boundary.accepted);

  const StationaryEvidence zero = evaluate(0.0, 21, 0.5, 0.0, 1.0, 0.0, 0.05);
  EXPECT_TRUE(zero.visual_stationary);
  EXPECT_TRUE(zero.inertial_stationary);
  EXPECT_TRUE(zero.accepted);
}

TEST(ZuptStationaryEvidence, RejectsInvalidDisparities) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<double> invalid_disparities = {nan, infinity, -infinity, -0.01};
  for (double disparity : invalid_disparities) {
    const StationaryEvidence result = evaluate(disparity, 100, 0.5, 0.0, 1.0, 0.0, 0.05);
    EXPECT_FALSE(result.visual_stationary);
    EXPECT_FALSE(result.accepted);
  }
}

TEST(ZuptStationaryEvidence, RejectsInvalidVisualThresholds) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<double> invalid_limits = {0.0, -0.5, nan, infinity, -infinity};
  for (double max_disparity : invalid_limits) {
    EXPECT_FALSE(evaluate(0.1, 100, max_disparity, 0.0, 1.0, 0.0, 0.05)
                     .visual_stationary);
  }
}

TEST(ZuptStationaryEvidence, RejectsInvalidInertialMeasurementsAndLimits) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<double> invalid_measurements = {nan, infinity, -infinity, -0.01};
  for (double value : invalid_measurements) {
    EXPECT_FALSE(evaluate(0.1, 100, 0.5, value, 1.0, 0.0, 0.05)
                     .inertial_stationary);
    EXPECT_FALSE(evaluate(0.1, 100, 0.5, 0.0, 1.0, value, 0.05)
                     .inertial_stationary);
  }

  const std::vector<double> invalid_limits = {0.0, -1.0, nan, infinity, -infinity};
  for (double value : invalid_limits) {
    EXPECT_FALSE(evaluate(0.1, 100, 0.5, 0.0, value, 0.0, 0.05)
                     .inertial_stationary);
    EXPECT_FALSE(evaluate(0.1, 100, 0.5, 0.0, 1.0, 0.0, value)
                     .inertial_stationary);
  }
}

}  // namespace

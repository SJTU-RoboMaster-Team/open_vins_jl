#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <limits>
#include "utils/instant_stereo.h"
#include "feat/FeatureDatabase.h"

namespace {
Eigen::Matrix3d camera() {
  Eigen::Matrix3d K;
  K << 800, 0, 319.5, 0, 550, 199.5, 0, 0, 1;
  return K;
}
Eigen::Vector2d project(const Eigen::Matrix3d &K, const Eigen::Vector3d &p) {
  return (K * p).head<2>() / p.z();
}

TEST(InstantStereo, RecoversMetricRangeFromOneFixedPoseWithoutMotionOrHistory) {
  const auto K = camera();
  const Eigen::Vector3d translation(-0.109, 0, 0);
  for (double depth : {0.5, 1.0, 2.0, 5.0, 10.0}) {
    const Eigen::Vector3d point(0.05, -0.03, depth);
    ov_core::InstantStereoPoint result;
    ASSERT_TRUE(ov_core::triangulate_instant_stereo(project(K, point), project(K, point + translation),
                                                 K, K, Eigen::Matrix3d::Identity(), translation, result));
    EXPECT_LT((result.position_c0 - point).norm(), 1e-8);
    EXPECT_LT(result.reprojection_error_px, 1e-8);
  }
}

TEST(InstantStereo, HandlesGeneralCalibratedRelativeRotationAndTranslationSign) {
  const auto K = camera();
  const Eigen::Matrix3d R = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  for (double sign : {-1.0, 1.0}) {
    const Eigen::Vector3d translation(sign * 0.109, 0.005, 0.002);
    const Eigen::Vector3d point(-0.1, 0.12, 2.1);
    ov_core::InstantStereoPoint result;
    ASSERT_TRUE(ov_core::triangulate_instant_stereo(project(K, point), project(K, R * point + translation),
                                                 K, K, R, translation, result));
    EXPECT_LT((result.position_c0 - point).norm(), 1e-9);
  }
}

TEST(InstantStereo, RejectsZeroParallaxBehindCameraAndInconsistentCorrespondence) {
  const auto K = camera();
  const Eigen::Vector3d translation(-0.109, 0, 0);
  ov_core::InstantStereoPoint result;
  EXPECT_FALSE(ov_core::triangulate_instant_stereo({320, 200}, {320, 200}, K, K,
                                                Eigen::Matrix3d::Identity(), translation, result));
  const Eigen::Vector3d behind(0, 0, -2);
  EXPECT_FALSE(ov_core::triangulate_instant_stereo(project(K, behind), project(K, behind + translation), K, K,
                                                Eigen::Matrix3d::Identity(), translation, result));
  EXPECT_FALSE(ov_core::triangulate_instant_stereo({320, 200}, {280, 220}, K, K,
                                                Eigen::Matrix3d::Identity(), translation, result));
  EXPECT_FALSE(ov_core::triangulate_instant_stereo({320, 200}, {280, 200}, K, K,
                                                Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), result));
}

TEST(InstantStereo, RejectsNonFiniteAndInvalidPhysicalGeometry) {
  const auto K = camera();
  const Eigen::Vector3d t(-0.109, 0, 0);
  ov_core::InstantStereoPoint result;
  Eigen::Vector2d bad(std::numeric_limits<double>::quiet_NaN(), 0);
  EXPECT_FALSE(ov_core::triangulate_instant_stereo(bad, {280, 200}, K, K, Eigen::Matrix3d::Identity(), t, result));
  EXPECT_FALSE(ov_core::triangulate_instant_stereo({320, 200}, {280, 200}, K, K, 2 * Eigen::Matrix3d::Identity(), t, result));
  EXPECT_FALSE(ov_core::triangulate_instant_stereo({320, 200}, {280, 200}, K, K, Eigen::Matrix3d::Identity(), t,
                                                result, 5, 1));
}

TEST(InstantStereo, RejectsPoorAngleAndPointsOutsideConfiguredRange) {
  const auto K = camera();
  const Eigen::Vector3d t(-0.109, 0, 0);
  ov_core::InstantStereoPoint result;
  const Eigen::Vector3d far(0, 0, 100);
  EXPECT_FALSE(ov_core::triangulate_instant_stereo(project(K, far), project(K, far + t), K, K,
                                                Eigen::Matrix3d::Identity(), t, result));
  const Eigen::Vector3d near(0, 0, 0.1);
  EXPECT_FALSE(ov_core::triangulate_instant_stereo(project(K, near), project(K, near + t), K, K,
                                                Eigen::Matrix3d::Identity(), t, result));
}

TEST(InstantStereo, DatabaseReturnsOnlyCurrentPairedIdsWithoutTemporalBaseline) {
  ov_core::FeatureDatabase database;
  database.update_feature(10, 1, 0, 320, 200, 0, 0);
  database.update_feature(10, 1, 1, 280, 200, 0, 0);
  database.update_feature(20, 1, 0, 330, 210, 0, 0);
  database.update_feature(21, 1, 1, 290, 210, 0, 0);
  database.update_feature(30, 1, 0, 340, 220, 0, 0);
  database.update_feature(30, 2, 1, 300, 220, 0, 0);
  const auto paired = database.latest_stereo_observations(1);
  ASSERT_EQ(paired.size(), 1u);
  EXPECT_EQ(paired[0].feature_id, 10u);
  EXPECT_EQ(paired[0].uv0.x(), 320);
  EXPECT_EQ(paired[0].uv1.x(), 280);
  database.update_feature(10, 2, 0, 321, 200, 0, 0);
  EXPECT_TRUE(database.latest_stereo_observations(1).empty());
  EXPECT_TRUE(database.latest_stereo_observations(2).empty());
  database.update_feature(10, 2, 1, 281, 200, 0, 0);
  EXPECT_EQ(database.latest_stereo_observations(2).size(), 1u);
  EXPECT_EQ(paired[0].uv0.x(), 320);
  EXPECT_TRUE(database.latest_stereo_observations(std::numeric_limits<double>::quiet_NaN()).empty());
}
} // namespace

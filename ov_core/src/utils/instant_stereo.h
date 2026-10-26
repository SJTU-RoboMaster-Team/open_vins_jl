/* Same-exposure calibrated sparse stereo geometry; no motion state is used. */
#ifndef OV_CORE_INSTANT_STEREO_H
#define OV_CORE_INSTANT_STEREO_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>

namespace ov_core {

struct InstantStereoPoint {
  Eigen::Vector3d position_c0 = Eigen::Vector3d::Zero();
  double reprojection_error_px = 0;
  double ray_angle_deg = 0;
};

inline bool triangulate_instant_stereo(
    const Eigen::Vector2d &uv0, const Eigen::Vector2d &uv1,
    const Eigen::Matrix3d &K0, const Eigen::Matrix3d &K1,
    const Eigen::Matrix3d &R_1_0, const Eigen::Vector3d &t_1_0,
    InstantStereoPoint &result, double min_range_m = 0.2, double max_range_m = 20.0,
    double min_ray_angle_deg = 0.5, double max_reprojection_error_px = 2.0) {
  if (!uv0.allFinite() || !uv1.allFinite() || !K0.allFinite() || !K1.allFinite() ||
      !R_1_0.allFinite() || !t_1_0.allFinite() ||
      K0(0, 0) <= 0 || K0(1, 1) <= 0 || K1(0, 0) <= 0 || K1(1, 1) <= 0 ||
      !std::isfinite(min_range_m) || !std::isfinite(max_range_m) ||
      !std::isfinite(min_ray_angle_deg) || !std::isfinite(max_reprojection_error_px) ||
      min_range_m < 0 || max_range_m <= min_range_m || min_ray_angle_deg <= 0 ||
      min_ray_angle_deg >= 90 || max_reprojection_error_px <= 0 || t_1_0.norm() < 1e-6)
    return false;
  if ((R_1_0.transpose() * R_1_0 - Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
      std::abs(R_1_0.determinant() - 1) > 1e-6)
    return false;
  // This helper is for the zero-distortion, zero-skew virtual pinhole views.
  const auto valid_K = [](const Eigen::Matrix3d &K) {
    return std::abs(K(0, 1)) < 1e-12 && std::abs(K(1, 0)) < 1e-12 &&
           std::abs(K(2, 0)) < 1e-12 && std::abs(K(2, 1)) < 1e-12 && std::abs(K(2, 2) - 1) < 1e-12;
  };
  if (!valid_K(K0) || !valid_K(K1))
    return false;
  Eigen::Vector3d d0((uv0.x() - K0(0, 2)) / K0(0, 0), (uv0.y() - K0(1, 2)) / K0(1, 1), 1);
  Eigen::Vector3d d1((uv1.x() - K1(0, 2)) / K1(0, 0), (uv1.y() - K1(1, 2)) / K1(1, 1), 1);
  d0.normalize();
  d1 = R_1_0.transpose() * d1.normalized();
  const double cosine = std::max(-1.0, std::min(1.0, d0.dot(d1)));
  const double angle_deg = std::acos(cosine) * (180.0 / std::acos(-1.0));
  const double denominator = 1 - cosine * cosine;
  if (angle_deg < min_ray_angle_deg || denominator < 1e-12)
    return false;

  const Eigen::Vector3d center1_in0 = -R_1_0.transpose() * t_1_0;
  const double along0 = d0.dot(center1_in0);
  const double along1 = d1.dot(center1_in0);
  const double range0 = (along0 - cosine * along1) / denominator;
  const double range1 = (cosine * along0 - along1) / denominator;
  if (range0 <= 0 || range1 <= 0)
    return false;
  // The midpoint of the two closest ray points tolerates small image noise.
  const Eigen::Vector3d point0 = 0.5 * (d0 * range0 + center1_in0 + d1 * range1);
  const Eigen::Vector3d point1 = R_1_0 * point0 + t_1_0;
  const double range = point0.norm();
  if (!point0.allFinite() || !point1.allFinite() || point0.z() <= 0 || point1.z() <= 0 ||
      range < min_range_m || range > max_range_m)
    return false;
  const Eigen::Vector2d projected0 = (K0 * point0).head<2>() / point0.z();
  const Eigen::Vector2d projected1 = (K1 * point1).head<2>() / point1.z();
  const double error = std::max((projected0 - uv0).norm(), (projected1 - uv1).norm());
  if (!std::isfinite(error) || error > max_reprojection_error_px)
    return false;
  result.position_c0 = point0;
  result.ray_angle_deg = angle_deg;
  result.reprojection_error_px = error;
  return true;
}

} // namespace ov_core
#endif

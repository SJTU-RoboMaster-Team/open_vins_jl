#ifndef OV_MSCKF_ZUPT_STATIONARY_EVIDENCE_H
#define OV_MSCKF_ZUPT_STATIONARY_EVIDENCE_H

#include <cmath>

namespace ov_msckf {

struct StationaryEvidence {
  bool visual_stationary;
  bool inertial_stationary;
  bool accepted;
};

inline StationaryEvidence evaluate_stationary_evidence(
    double disparity, int feature_count, double max_disparity, double chi2,
    double chi2_limit, double speed, double max_speed) {
  const bool visual_stationary = std::isfinite(disparity) && disparity >= 0.0 &&
      std::isfinite(max_disparity) && max_disparity > 0.0 &&
      feature_count > 20 && disparity < max_disparity;
  const bool inertial_stationary = std::isfinite(chi2) && chi2 >= 0.0 &&
      std::isfinite(speed) && speed >= 0.0 &&
      std::isfinite(chi2_limit) && chi2_limit > 0.0 && chi2 <= chi2_limit &&
      std::isfinite(max_speed) && max_speed > 0.0 && speed <= max_speed;
  return {visual_stationary, inertial_stationary,
          visual_stationary && inertial_stationary};
}

}  // namespace ov_msckf

#endif  // OV_MSCKF_ZUPT_STATIONARY_EVIDENCE_H

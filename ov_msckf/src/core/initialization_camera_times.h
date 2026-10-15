#ifndef OV_MSCKF_INITIALIZATION_CAMERA_TIMES_H
#define OV_MSCKF_INITIALIZATION_CAMERA_TIMES_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace ov_msckf {

inline std::vector<double> independent_camera_times(
    const std::vector<double> &queued, double initialized_time) {
  std::vector<double> times;
  if (!std::isfinite(initialized_time)) return times;

  times.reserve(queued.size());
  for (double timestamp : queued) {
    if (std::isfinite(timestamp) && timestamp > initialized_time)
      times.push_back(timestamp);
  }

  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  return times;
}

}  // namespace ov_msckf

#endif  // OV_MSCKF_INITIALIZATION_CAMERA_TIMES_H

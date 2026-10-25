// Transport backlog policy; no ROS/OpenCV dependencies. Queue must be sorted.
#ifndef OV_MSCKF_CAMERA_BACKLOG_H
#define OV_MSCKF_CAMERA_BACKLOG_H
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ov_msckf {
// Discard synchronized time groups together, never a single camera from a group.
// A cutoff is retained so late-arriving members cannot resurrect a dropped group.
template <class Queue>
std::size_t trim_camera_backlog(Queue &queue, std::size_t max_messages,
                                double max_span, double sync_tolerance,
                                double &dropped_through) {
  std::size_t dropped = 0;
  while (!queue.empty() &&
         ((max_messages > 0 && queue.size() > max_messages) ||
          (max_span > 0 && queue.back().timestamp - queue.front().timestamp > max_span + 1e-6))) {
    const double group_end = queue.front().timestamp + std::max(0.0, sync_tolerance);
    dropped_through = std::max(dropped_through, group_end);
    do {
      queue.pop_front();
      ++dropped;
    } while (!queue.empty() && queue.front().timestamp <= group_end);
  }
  return dropped;
}
} // namespace ov_msckf
#endif

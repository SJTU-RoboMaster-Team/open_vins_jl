/*
 * Copyright (c) 2026.
 *
 * Minimal camera-bundle wait predicate. This header intentionally has no ROS,
 * Eigen, or OpenCV dependencies so the ordering rule can be tested standalone.
 */
#ifndef OV_MSCKF_UTILS_CAMERA_SYNC_H
#define OV_MSCKF_UTILS_CAMERA_SYNC_H

#include <cstddef>
#include <map>

namespace ov_msckf {

/**
 * Return whether an incomplete camera group might still receive a member that
 * belongs to the current synchronization window.
 *
 * Camera IDs are expected to be the contiguous range [0, expected_cameras).
 * A missing camera may still join unless its latest queued timestamp is
 * strictly newer than group_end. Missing queue entries and timestamps at or
 * before group_end therefore keep the caller waiting. This predicate only
 * reasons about camera queues; it deliberately does not use an IMU cutoff.
 *
 * A complete group never needs to wait for another camera member and returns
 * false. The function does not decide whether to discard stale frames or how
 * to consume a partial group; it only answers whether a missing synchronized
 * frame can still arrive.
 */
inline bool camera_bundle_wait_for_missing(
    std::size_t expected_cameras,
    const std::map<int, std::size_t> &group_members,
    const std::map<int, double> &latest_queued_stamp,
    double group_end) {
  if (expected_cameras == 0) {
    return false;
  }

  for (std::size_t camera = 0; camera < expected_cameras; ++camera) {
    const int camera_id = static_cast<int>(camera);
    if (group_members.find(camera_id) != group_members.end()) {
      continue;
    }

    const std::map<int, double>::const_iterator latest =
        latest_queued_stamp.find(camera_id);
    if (latest == latest_queued_stamp.end() || !(latest->second > group_end)) {
      return true;
    }
  }

  // A complete group returns false, as does a group whose every absent camera
  // has already advanced beyond this synchronization window.
  return false;
}

}  // namespace ov_msckf

#endif  // OV_MSCKF_UTILS_CAMERA_SYNC_H

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/initialization_camera_times.h"

namespace {

using ov_msckf::independent_camera_times;

TEST(InitializationCameraTimes, RemovesFourCameraRepeatedSamples) {
  const std::vector<double> queued = {
      10.0, 10.1, 10.2, 10.3,  // camera 0
      10.0, 10.1, 10.2, 10.3,  // camera 1
      10.0, 10.1, 10.2, 10.3,  // camera 2
      10.0, 10.1, 10.2, 10.3,  // camera 3
  };

  const std::vector<double> times = independent_camera_times(queued, 10.0);
  ASSERT_EQ(times.size(), 3U);
  EXPECT_DOUBLE_EQ(times[0], 10.1);
  EXPECT_DOUBLE_EQ(times[1], 10.2);
  EXPECT_DOUBLE_EQ(times[2], 10.3);
  EXPECT_LT(times[0], times[1]);
  EXPECT_LT(times[1], times[2]);
}

TEST(InitializationCameraTimes, SortsUnorderedQueueChronologically) {
  const std::vector<double> times =
      independent_camera_times({3.0, 1.0, 2.0, 1.0}, 0.0);
  ASSERT_EQ(times.size(), 3U);
  EXPECT_DOUBLE_EQ(times[0], 1.0);
  EXPECT_DOUBLE_EQ(times[1], 2.0);
  EXPECT_DOUBLE_EQ(times[2], 3.0);
}

TEST(InitializationCameraTimes, ExcludesSamplesAtOrBeforeInitialization) {
  const std::vector<double> times =
      independent_camera_times({1.0, 1.5, 2.0, 0.5}, 1.0);
  ASSERT_EQ(times.size(), 2U);
  EXPECT_DOUBLE_EQ(times[0], 1.5);
  EXPECT_DOUBLE_EQ(times[1], 2.0);
}

TEST(InitializationCameraTimes, ExcludesNonFiniteQueuedTimes) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<double> times = independent_camera_times(
      {nan, -infinity, 2.0, infinity, 1.0}, 0.0);
  ASSERT_EQ(times.size(), 2U);
  EXPECT_DOUBLE_EQ(times[0], 1.0);
  EXPECT_DOUBLE_EQ(times[1], 2.0);
}

TEST(InitializationCameraTimes, EmptyQueueReturnsEmpty) {
  EXPECT_TRUE(independent_camera_times({}, 0.0).empty());
}

TEST(InitializationCameraTimes, NonFiniteInitializationTimeReturnsEmpty) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(independent_camera_times({1.0, 2.0}, nan).empty());
  EXPECT_TRUE(independent_camera_times({1.0, 2.0}, infinity).empty());
  EXPECT_TRUE(independent_camera_times({1.0, 2.0}, -infinity).empty());
}

TEST(InitializationCameraTimes, KeepsCloseButDistinctTimestamps) {
  const double first = 1.0;
  const double second = std::nextafter(first, 2.0);
  ASSERT_GT(second, first);

  const std::vector<double> times =
      independent_camera_times({second, first}, 0.0);
  ASSERT_EQ(times.size(), 2U);
  EXPECT_DOUBLE_EQ(times[0], first);
  EXPECT_DOUBLE_EQ(times[1], second);
}

TEST(InitializationCameraTimes, UniqueChronologicalTimesSupportCloneStride) {
  const std::vector<double> queued = {
      4.0, 4.1, 4.2, 4.3,  // camera 0
      4.0, 4.1, 4.2, 4.3,  // camera 1
      4.0, 4.1, 4.2, 4.3,  // camera 2
      4.0, 4.1, 4.2, 4.3,  // camera 3
  };
  const std::vector<double> times = independent_camera_times(queued, 3.9);
  ASSERT_EQ(times.size(), 4U);

  const std::size_t clone_stride = 2U;
  std::vector<double> clone_times;
  for (std::size_t index = 0; index < times.size(); index += clone_stride)
    clone_times.push_back(times[index]);

  ASSERT_EQ(clone_times.size(), 2U);
  EXPECT_DOUBLE_EQ(clone_times[0], 4.0);
  EXPECT_DOUBLE_EQ(clone_times[1], 4.2);
  EXPECT_LT(clone_times[0], clone_times[1]);
}

}  // namespace

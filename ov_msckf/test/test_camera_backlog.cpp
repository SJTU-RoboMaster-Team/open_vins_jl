#include "utils/camera_backlog.h"
#include "utils/camera_sync.h"
#include <gtest/gtest.h>
#include <deque>
#include <map>

namespace {
struct Frame { double timestamp; int camera; };
void group(std::deque<Frame> &q, double t) {
  for (int c = 0; c < 4; ++c) q.push_back({t, c});
}
TEST(CameraBacklog, KeepsNormalFourCameraStream) {
  std::deque<Frame> q;
  for (int i = 0; i < 4; ++i) group(q, 1792821807.0 + i * .05);
  double cutoff = -1;
  EXPECT_EQ(0u, ov_msckf::trim_camera_backlog(q, 16, .15, .001, cutoff));
  EXPECT_EQ(16u, q.size());
  EXPECT_EQ(-1, cutoff);
}
TEST(CameraBacklog, CountLimitDiscardsWholeGroups) {
  std::deque<Frame> q;
  for (int i = 0; i < 6; ++i) group(q, i * .05);
  double cutoff = -1;
  EXPECT_EQ(8u, ov_msckf::trim_camera_backlog(q, 16, 1, .001, cutoff));
  EXPECT_EQ(16u, q.size());
  EXPECT_EQ(0, q.front().camera);
  EXPECT_DOUBLE_EQ(.1, q.front().timestamp);
  EXPECT_NEAR(.051, cutoff, 1e-9);
}
TEST(CameraBacklog, AgeLimitDiscardsWholeGroups) {
  std::deque<Frame> q;
  for (int i = 0; i < 8; ++i) group(q, i * .05);
  double cutoff = -1;
  EXPECT_EQ(16u, ov_msckf::trim_camera_backlog(q, 100, .15, .001, cutoff));
  EXPECT_EQ(16u, q.size());
  EXPECT_NEAR(.2, q.front().timestamp, 1e-9);
}
TEST(CameraBacklog, NearSynchronousGroupIsNotSplit) {
  std::deque<Frame> q{{1,0},{1.0003,1},{1.0006,2},{1.0009,3},{1.05,0}};
  double cutoff = -1;
  EXPECT_EQ(4u, ov_msckf::trim_camera_backlog(q, 4, 1, .001, cutoff));
  ASSERT_EQ(1u,q.size());
  EXPECT_DOUBLE_EQ(1.05,q.front().timestamp);
  EXPECT_LE(1.0009,cutoff); // a late fourth member is also rejected by ingress
}
TEST(CameraBacklog, MissingCameraCannotGrowQueueForever) {
  std::deque<Frame> q;
  double cutoff = -1;
  for (int i = 0; i < 1000; ++i) {
    for (int c = 0; c < 3; ++c) q.push_back({i*.05,c});
    ov_msckf::trim_camera_backlog(q,16,.15,.001,cutoff);
    EXPECT_LE(q.size(),16u);
    EXPECT_LE(q.back().timestamp-q.front().timestamp,.150001);
  }
}
TEST(CameraBacklog, SyncStillWaitsForInFlightMemberDespiteNewIMU) {
  std::map<int,size_t> members{{0,0},{1,1},{2,2}};
  std::map<int,double> latest{{0,1},{1,1},{2,1}};
  EXPECT_TRUE(ov_msckf::camera_bundle_wait_for_missing(4,members,latest,1.001));
  latest[3]=1.05;
  EXPECT_FALSE(ov_msckf::camera_bundle_wait_for_missing(4,members,latest,1.001));
}
}

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "core/VioManagerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {

using ov_core::ImuData;
using ov_msckf::ImuSelectionDiagnostics;
using ov_msckf::NoiseManager;
using ov_msckf::Propagator;
using ov_msckf::State;
using ov_msckf::StateOptions;
using SteadyClock = std::chrono::steady_clock;

ImuData sample(double timestamp, double gx, double gy, double gz,
               double ax, double ay, double az) {
  ImuData data;
  data.timestamp = timestamp;
  data.wm << gx, gy, gz;
  data.am << ax, ay, az;
  return data;
}

bool same_measurements(const std::vector<ImuData> &left,
                      const std::vector<ImuData> &right) {
  if (left.size() != right.size()) return false;
  for (size_t index = 0; index < left.size(); ++index) {
    if (left[index].timestamp != right[index].timestamp ||
        !(left[index].wm.array() == right[index].wm.array()).all() ||
        !(left[index].am.array() == right[index].am.array()).all()) {
      return false;
    }
  }
  return true;
}

std::vector<ImuData> select_with_diagnostics(const std::vector<ImuData> &input,
                                            double start, double end,
                                            ImuSelectionDiagnostics *diagnostics) {
  return Propagator::select_imu_readings(input, start, end, false, diagnostics);
}

void expect_sample(const ImuData &actual, double timestamp,
                   const Eigen::Vector3d &gyro, const Eigen::Vector3d &accel) {
  EXPECT_NEAR(actual.timestamp, timestamp, 1e-12);
  EXPECT_TRUE(actual.wm.isApprox(gyro, 1e-12));
  EXPECT_TRUE(actual.am.isApprox(accel, 1e-12));
}

bool write_all(int fd, const void *bytes, size_t size) {
  const char *data = static_cast<const char *>(bytes);
  size_t written = 0;
  while (written < size) {
    const ssize_t count = write(fd, data + written, size - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    written += static_cast<size_t>(count);
  }
  return true;
}

bool fill_pipe_without_blocking(int fd) {
  const int original = fcntl(fd, F_GETFL);
  if (original < 0 || fcntl(fd, F_SETFL, original | O_NONBLOCK) < 0) return false;
  std::array<char, 4096> padding{};
  bool full = false;
  while (true) {
    const ssize_t count = write(fd, padding.data(), padding.size());
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      full = true;
      break;
    }
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) break;
  }
  if (fcntl(fd, F_SETFL, original) < 0) return false;
  return full;
}

bool poll_read(int fd, void *buffer, size_t size, int timeout_ms) {
  struct pollfd descriptor;
  descriptor.fd = fd;
  descriptor.events = POLLIN | POLLHUP;
  descriptor.revents = 0;
  int ready;
  do {
    ready = poll(&descriptor, 1, timeout_ms);
  } while (ready < 0 && errno == EINTR);
  if (ready <= 0 || !(descriptor.revents & (POLLIN | POLLHUP))) return false;
  size_t received = 0;
  char *bytes = static_cast<char *>(buffer);
  while (received < size) {
    const ssize_t count = read(fd, bytes + received, size - received);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    received += static_cast<size_t>(count);
  }
  return true;
}

int drain_and_reap(pid_t child, int output_fd) {
  const int flags = fcntl(output_fd, F_GETFL);
  if (flags >= 0) fcntl(output_fd, F_SETFL, flags | O_NONBLOCK);
  const auto deadline = SteadyClock::now() + std::chrono::seconds(3);
  std::array<char, 8192> discard{};
  bool exited = false;
  int child_status = 0;
  while (SteadyClock::now() < deadline) {
    while (read(output_fd, discard.data(), discard.size()) > 0) {}
    int status = 0;
    const pid_t result = waitpid(child, &status, WNOHANG);
    if (result == child) {
      exited = true;
      child_status = status;
      break;
    }
    if (result < 0 && errno == ECHILD) {
      exited = true;
      break;
    }
    usleep(10000);
  }
  if (!exited) {
    kill(child, SIGKILL);
    waitpid(child, &child_status, 0);
  }
  close(output_fd);
  return child_status;
}

}  // namespace

TEST(PropagatorSelectionDiagnostics, PreserveSelectedMeasurementsAndReportCases) {
  const std::vector<ImuData> bracketing = {
      sample(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 9.81),
      sample(0.1, 2.0, 0.0, 1.0, 2.0, 4.0, 9.81),
      sample(0.2, 4.0, -2.0, 2.0, 4.0, 8.0, 9.81),
  };
  ImuSelectionDiagnostics interpolation_diagnostics;
  const auto interpolated = select_with_diagnostics(
      bracketing, 0.05, 0.15, &interpolation_diagnostics);
  const auto interpolated_without_stats =
      Propagator::select_imu_readings(bracketing, 0.05, 0.15, false);
  EXPECT_TRUE(same_measurements(interpolated, interpolated_without_stats));
  ASSERT_EQ(interpolated.size(), 3U);
  expect_sample(interpolated[0], 0.05, Eigen::Vector3d(1.0, 1.0, 0.5),
                Eigen::Vector3d(1.0, 2.0, 9.81));
  expect_sample(interpolated[1], 0.1, Eigen::Vector3d(2.0, 0.0, 1.0),
                Eigen::Vector3d(2.0, 4.0, 9.81));
  expect_sample(interpolated[2], 0.15, Eigen::Vector3d(3.0, -1.0, 1.5),
                Eigen::Vector3d(3.0, 6.0, 9.81));
  EXPECT_FALSE(interpolation_diagnostics.no_input);
  EXPECT_FALSE(interpolation_diagnostics.no_interval);
  EXPECT_FALSE(interpolation_diagnostics.insufficient);
  EXPECT_EQ(interpolation_diagnostics.zero_dt_removed_count, 0U);

  const std::vector<ImuData> duplicates = {
      sample(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 9.81),
      sample(0.05, 1.0, 1.0, 0.5, 1.0, 2.0, 9.81),
      sample(0.1, 2.0, 0.0, 1.0, 2.0, 4.0, 9.81),
      sample(0.1, 2.0, 0.0, 1.0, 2.0, 4.0, 9.81),
      sample(0.15, 3.0, -1.0, 1.5, 3.0, 6.0, 9.81),
      sample(0.2, 4.0, -2.0, 2.0, 4.0, 8.0, 9.81),
  };
  ImuSelectionDiagnostics duplicate_diagnostics;
  const auto deduplicated = select_with_diagnostics(
      duplicates, 0.05, 0.15, &duplicate_diagnostics);
  const auto deduplicated_without_stats =
      Propagator::select_imu_readings(duplicates, 0.05, 0.15, false);
  EXPECT_TRUE(same_measurements(deduplicated, deduplicated_without_stats));
  ASSERT_EQ(deduplicated.size(), 3U);
  expect_sample(deduplicated[0], 0.05, Eigen::Vector3d(1.0, 1.0, 0.5),
                Eigen::Vector3d(1.0, 2.0, 9.81));
  expect_sample(deduplicated[1], 0.1, Eigen::Vector3d(2.0, 0.0, 1.0),
                Eigen::Vector3d(2.0, 4.0, 9.81));
  expect_sample(deduplicated[2], 0.15, Eigen::Vector3d(3.0, -1.0, 1.5),
                Eigen::Vector3d(3.0, 6.0, 9.81));
  EXPECT_EQ(duplicate_diagnostics.zero_dt_removed_count, 1U);
  EXPECT_FALSE(duplicate_diagnostics.insufficient);

  ImuSelectionDiagnostics empty_diagnostics;
  EXPECT_TRUE(select_with_diagnostics({}, 0.0, 1.0, &empty_diagnostics).empty());
  EXPECT_TRUE(empty_diagnostics.no_input);

  ImuSelectionDiagnostics interval_diagnostics;
  EXPECT_TRUE(select_with_diagnostics(bracketing, 0.3, 0.4,
                                      &interval_diagnostics).empty());
  EXPECT_TRUE(interval_diagnostics.no_interval);

  const std::vector<ImuData> single_selected = {
      sample(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 9.81),
      sample(0.1, 2.0, 0.0, 1.0, 2.0, 4.0, 9.81),
      sample(0.1, 2.0, 0.0, 1.0, 2.0, 4.0, 9.81),
  };
  ImuSelectionDiagnostics insufficient_diagnostics;
  const auto insufficient = select_with_diagnostics(
      single_selected, 0.1, 0.1, &insufficient_diagnostics);
  ASSERT_EQ(insufficient.size(), 1U);
  expect_sample(insufficient.front(), 0.1, Eigen::Vector3d(2.0, 0.0, 1.0),
                Eigen::Vector3d(2.0, 4.0, 9.81));
  EXPECT_TRUE(insufficient_diagnostics.insufficient);
  EXPECT_EQ(insufficient_diagnostics.selected_count, 1U);
}

TEST(PropagatorDiagnostics, SlowStdoutDoesNotHoldImuBufferLock) {
  const auto old_print_level = ov_core::Printer::current_print_level;
  ov_core::Printer::setPrintLevel(ov_core::Printer::PrintLevel::WARNING);

  int output_pipe[2];
  int status_pipe[2];
  ASSERT_EQ(pipe(output_pipe), 0);
  ASSERT_EQ(pipe(status_pipe), 0);
  ASSERT_TRUE(fill_pipe_without_blocking(output_pipe[1]));

  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(output_pipe[0]);
    close(status_pipe[0]);
    if (dup2(output_pipe[1], STDOUT_FILENO) < 0) _exit(10);
    close(output_pipe[1]);
    setvbuf(stdout, nullptr, _IONBF, 0);

    StateOptions options;
    auto state = std::make_shared<State>(options);
    state->_timestamp = 0.1;
    NoiseManager noises;
    Propagator propagator(noises, 9.81);
    const std::array<double, 7> stamps = {0.0, 0.05, 0.1, 0.15, 0.15, 0.2, 0.25};
    for (double stamp : stamps) {
      propagator.feed_imu(sample(stamp, 0.0, 0.0, 0.0, 0.0, 0.0, 9.81));
    }

    std::thread propagation([&] { propagator.propagate_and_clone(state, 0.2); });
    usleep(100000);
    const auto feed_started = SteadyClock::now();
    propagator.feed_imu(sample(0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 9.81));
    const auto feed_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        SteadyClock::now() - feed_started).count();
    const int64_t feed_time_us = static_cast<int64_t>(feed_elapsed);
    const bool reported = write_all(status_pipe[1], &feed_time_us, sizeof(feed_time_us));
    propagation.join();
    close(status_pipe[1]);
    _exit(reported ? 0 : 11);
  }

  close(output_pipe[1]);
  close(status_pipe[1]);
  int64_t feed_time_us = -1;
  bool reported_before_drain =
      poll_read(status_pipe[0], &feed_time_us, sizeof(feed_time_us), 300);
  const int child_status = drain_and_reap(child, output_pipe[0]);
  if (!reported_before_drain) {
    reported_before_drain =
        poll_read(status_pipe[0], &feed_time_us, sizeof(feed_time_us), 1000);
  }
  close(status_pipe[0]);
  ov_core::Printer::setPrintLevel(old_print_level);

  ASSERT_TRUE(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0)
      << "isolated propagation child failed, status=" << child_status;
  ASSERT_TRUE(reported_before_drain)
      << "feed_imu did not complete while propagate_and_clone was blocked "
         "writing its duplicate-sample diagnostic";
  EXPECT_LE(feed_time_us, 200000)
      << "feed_imu waited " << feed_time_us
      << " us while the propagation diagnostic was blocked on stdout";
}


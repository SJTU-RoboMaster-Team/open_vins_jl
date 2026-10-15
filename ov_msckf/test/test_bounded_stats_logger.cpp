#include <gtest/gtest.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <stdlib.h>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "core/bounded_stats_logger.h"

namespace {

class TempFile {
public:
  TempFile() {
    char path[] = "/tmp/ov_msckf_stats_XXXXXX";
    fd = ::mkstemp(path);
    if (fd >= 0) path_ = path;
  }
  ~TempFile() {
    if (fd >= 0) ::close(fd);
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;

  std::string read_all() const {
    if (fd < 0) return {};
    struct stat status;
    if (::fstat(fd, &status) != 0 || status.st_size < 0) return {};
    if (status.st_size == 0) return {};
    std::string contents(static_cast<std::size_t>(status.st_size), '\0');
    if (::pread(fd, &contents[0], contents.size(), 0) != static_cast<ssize_t>(contents.size())) return {};
    return contents;
  }

  int fd = -1;

private:
  std::string path_;
};

bool wait_until(const std::function<bool()> &predicate, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::yield();
  }
  return predicate();
}

} // namespace

TEST(BoundedStatsLogger, PreservesRegularFileOffsetAndOriginalFlags) {
  TempFile output;
  ASSERT_GE(output.fd, 0);
  const int flags_before = ::fcntl(output.fd, F_GETFL);
  ASSERT_GE(flags_before, 0);

  ASSERT_EQ(::write(output.fd, "before\n", 7), 7);
  {
    ov_msckf::BoundedStatsLogger logger(output.fd);
    ASSERT_TRUE(logger.try_enqueue("logger\n", 7));
    ASSERT_TRUE(wait_until([&] { return logger.counters().written == 1; }, std::chrono::milliseconds(500)));
    EXPECT_EQ(::fcntl(output.fd, F_GETFL), flags_before);
    ASSERT_EQ(::write(output.fd, "after\n", 6), 6);
    logger.stop();
    logger.stop();
  }

  EXPECT_EQ(output.read_all(), "before\nlogger\nafter\n");
  EXPECT_EQ(::fcntl(output.fd, F_GETFL), flags_before);
}

TEST(BoundedStatsLogger, InstancesWriteIndependentlyAndSnapshotsAreCopies) {
  TempFile first;
  TempFile second;
  ASSERT_GE(first.fd, 0);
  ASSERT_GE(second.fd, 0);
  ov_msckf::BoundedStatsLogger logger_a(first.fd);
  ov_msckf::BoundedStatsLogger logger_b(second.fd);
  ASSERT_TRUE(logger_a.try_enqueue("first\n", 6));
  ASSERT_TRUE(wait_until([&] { return logger_a.counters().written == 1; }, std::chrono::milliseconds(500)));
  const auto snapshot = logger_a.counters();
  ASSERT_TRUE(logger_b.try_enqueue("second\n", 7));
  ASSERT_TRUE(wait_until([&] { return logger_b.counters().written == 1; }, std::chrono::milliseconds(500)));
  ASSERT_TRUE(logger_a.try_enqueue("again\n", 6));
  ASSERT_TRUE(wait_until([&] { return logger_a.counters().written == 2; }, std::chrono::milliseconds(500)));
  logger_a.stop();
  logger_b.stop();

  EXPECT_EQ(first.read_all(), "first\nagain\n");
  EXPECT_EQ(second.read_all(), "second\n");
  EXPECT_EQ(snapshot.enqueued, 1u);
  EXPECT_EQ(snapshot.written, 1u);
  EXPECT_EQ(snapshot.queued, 0u);
  EXPECT_EQ(snapshot.enqueued, logger_a.counters().enqueued - 1);
  EXPECT_EQ(snapshot.written, logger_a.counters().written - 1);
}

TEST(BoundedStatsLogger, CopiesPayloadAtEnqueue) {
  TempFile output;
  ASSERT_GE(output.fd, 0);
  char payload[] = "alpha\n";
  ov_msckf::BoundedStatsLogger logger(output.fd);
  ASSERT_TRUE(logger.try_enqueue(payload, sizeof(payload) - 1));
  std::memcpy(payload, "bravo\n", sizeof(payload) - 1);
  ASSERT_TRUE(wait_until([&] { return logger.counters().written == 1; }, std::chrono::milliseconds(500)));
  logger.stop();
  EXPECT_EQ(output.read_all(), "alpha\n");
}

TEST(BoundedStatsLogger, TruncatesRecordsAtFixedSlotSize) {
  TempFile output;
  ASSERT_GE(output.fd, 0);
  std::vector<char> large(ov_msckf::BoundedStatsLogger::kMaxRecordBytes + 99, 'x');
  ov_msckf::BoundedStatsLogger logger(output.fd);
  ASSERT_TRUE(logger.try_enqueue(large.data(), large.size()));
  ASSERT_TRUE(wait_until([&] { return logger.counters().written == 1; }, std::chrono::milliseconds(500)));
  ASSERT_TRUE(wait_until([&] {
    return output.read_all().find("[BoundedStatsLogger]") != std::string::npos;
  }, std::chrono::milliseconds(500)));
  logger.stop();

  EXPECT_EQ(logger.counters().truncated, 1u);
  EXPECT_EQ(logger.counters().written, 1u);
  const std::string contents = output.read_all();
  const std::size_t record_bytes = static_cast<std::size_t>(ov_msckf::BoundedStatsLogger::kMaxRecordBytes);
  ASSERT_GE(contents.size(), record_bytes);
  EXPECT_EQ(contents.substr(0, record_bytes), std::string(record_bytes, 'x'));
  EXPECT_NE(contents.find("[BoundedStatsLogger]"), std::string::npos);
  EXPECT_NE(contents.find("truncated=1"), std::string::npos);
}

TEST(BoundedStatsLogger, EmptyRecordIsNoOp) {
  TempFile output;
  ASSERT_GE(output.fd, 0);
  ov_msckf::BoundedStatsLogger logger(output.fd);
  EXPECT_TRUE(logger.try_enqueue(nullptr, 0));
  EXPECT_EQ(logger.counters().enqueued, 0u);
  EXPECT_EQ(logger.counters().queued, 0u);
  logger.stop();
  EXPECT_TRUE(output.read_all().empty());
}

TEST(BoundedStatsLogger, FullPipeDoesNotBlockProducersAndStopIsBounded) {
  int pipe_fds[2];
  ASSERT_EQ(::pipe(pipe_fds), 0);
  const int original_flags = ::fcntl(pipe_fds[1], F_GETFL);
  ASSERT_GE(original_flags, 0);
  ASSERT_EQ(::fcntl(pipe_fds[1], F_SETFL, original_flags | O_NONBLOCK), 0);
  const int expected_flags = ::fcntl(pipe_fds[1], F_GETFL);
  ASSERT_GE(expected_flags, 0);
  std::array<char, 4096> filler{};
  filler.fill('p');
  while (::write(pipe_fds[1], filler.data(), filler.size()) > 0) {}
  ASSERT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);

  ov_msckf::BoundedStatsLogger logger(pipe_fds[1]);
  const char record[] = "record\n";
  const auto before_calls = logger.counters();
  std::uint64_t accepted = 0;
  for (int i = 0; i < 64; ++i) {
    if (logger.try_enqueue(record, sizeof(record) - 1)) ++accepted;
  }
  const auto after_calls = logger.counters();
  const std::uint64_t enqueued_delta = after_calls.enqueued - before_calls.enqueued;
  const std::uint64_t full_delta = after_calls.dropped_full - before_calls.dropped_full;
  const std::uint64_t contended_delta = after_calls.dropped_contended - before_calls.dropped_contended;
  const std::uint64_t sink_delta = after_calls.dropped_sink - before_calls.dropped_sink;
  EXPECT_EQ(enqueued_delta, accepted);
  EXPECT_EQ(full_delta + contended_delta, 64u - accepted);
  EXPECT_EQ(sink_delta, 0u);
  EXPECT_LE(after_calls.queued, static_cast<std::size_t>(ov_msckf::BoundedStatsLogger::kSlotCount));
  EXPECT_GT(full_delta + contended_delta, 0u);
  EXPECT_EQ(::fcntl(pipe_fds[1], F_GETFL), expected_flags);

  const auto stop_start = std::chrono::steady_clock::now();
  logger.stop();
  EXPECT_LT(std::chrono::steady_clock::now() - stop_start, std::chrono::milliseconds(500));
  const auto final = logger.counters();
  EXPECT_EQ(final.queued, 0u);
  EXPECT_EQ(final.enqueued, final.written + final.dropped_sink);
  EXPECT_EQ(::fcntl(pipe_fds[1], F_GETFL), expected_flags);
  ::close(pipe_fds[0]);
  ::close(pipe_fds[1]);
}

TEST(BoundedStatsLogger, ReportsDropsAfterPipeRecoversAndClosedReaderIsSafe) {
  int pipe_fds[2];
  ASSERT_EQ(::pipe(pipe_fds), 0);
  const int flags = ::fcntl(pipe_fds[1], F_GETFL);
  ASSERT_GE(flags, 0);
  ASSERT_EQ(::fcntl(pipe_fds[1], F_SETFL, flags | O_NONBLOCK), 0);
  ASSERT_EQ(::fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK), 0);
  std::array<char, 4096> filler{};
  filler.fill('p');
  while (::write(pipe_fds[1], filler.data(), filler.size()) > 0) {}

  ov_msckf::BoundedStatsLogger logger(pipe_fds[1]);
  const char record[] = "r\n";
  const auto before_calls = logger.counters();
  for (int i = 0; i < 64; ++i) (void)logger.try_enqueue(record, sizeof(record) - 1);
  const auto drop_snapshot = logger.counters();
  ASSERT_GT((drop_snapshot.dropped_full - before_calls.dropped_full) +
                (drop_snapshot.dropped_contended - before_calls.dropped_contended), 0u);

  std::array<char, 8192> drain{};
  while (::read(pipe_fds[0], drain.data(), drain.size()) > 0) {}
  std::string received;
  ASSERT_TRUE(wait_until([&] {
    const ssize_t count = ::read(pipe_fds[0], drain.data(), drain.size());
    if (count > 0) received.append(drain.data(), static_cast<std::size_t>(count));
    return received.find("[BoundedStatsLogger] dropped_full=") != std::string::npos;
  }, std::chrono::milliseconds(1000)));
  EXPECT_NE(received.find("[BoundedStatsLogger]"), std::string::npos);
  const auto summary_pos = received.find("[BoundedStatsLogger] dropped_full=");
  unsigned long long summary_full = 0, summary_contended = 0, summary_sink = 0, summary_truncated = 0;
  ASSERT_EQ(std::sscanf(received.c_str() + summary_pos,
      "[BoundedStatsLogger] dropped_full=%llu dropped_contended=%llu dropped_sink=%llu truncated=%llu",
      &summary_full, &summary_contended, &summary_sink, &summary_truncated), 4);
  EXPECT_GE(summary_full, drop_snapshot.dropped_full);
  EXPECT_GE(summary_contended, drop_snapshot.dropped_contended);
  EXPECT_GE(summary_sink, drop_snapshot.dropped_sink);
  logger.stop();
  ::close(pipe_fds[0]);
  ::close(pipe_fds[1]);

  int closed_pipe[2];
  ASSERT_EQ(::pipe(closed_pipe), 0);
  struct sigaction sigpipe_before;
  ASSERT_EQ(::sigaction(SIGPIPE, nullptr, &sigpipe_before), 0);
  {
    ov_msckf::BoundedStatsLogger closed_logger(closed_pipe[1]);
    ASSERT_TRUE(closed_logger.try_enqueue(record, sizeof(record) - 1));
    ASSERT_TRUE(wait_until([&] { return closed_logger.counters().written == 1; },
                           std::chrono::milliseconds(500)));
    std::array<char, sizeof(record)> probe{};
    ASSERT_EQ(::read(closed_pipe[0], probe.data(), sizeof(record) - 1),
              static_cast<ssize_t>(sizeof(record) - 1));

    // The sink is known open now; closing its only reader forces a real EPIPE
    // write in the worker instead of merely testing an open-time ENXIO error.
    ::close(closed_pipe[0]);
    ASSERT_TRUE(closed_logger.try_enqueue(record, sizeof(record) - 1));
    ASSERT_TRUE(wait_until([&] { return closed_logger.counters().dropped_sink >= 1; },
                           std::chrono::milliseconds(500)));
    closed_logger.stop();
    EXPECT_GE(closed_logger.counters().dropped_sink, 1u);
    SUCCEED(); // The process survived the worker's EPIPE.
  }
  struct sigaction sigpipe_after;
  ASSERT_EQ(::sigaction(SIGPIPE, nullptr, &sigpipe_after), 0);
  EXPECT_EQ(sigpipe_after.sa_handler, sigpipe_before.sa_handler);
  ::close(closed_pipe[1]);
}


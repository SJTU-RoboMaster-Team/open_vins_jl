#ifndef OV_MSCKF_BOUNDED_STATS_LOGGER_H
#define OV_MSCKF_BOUNDED_STATS_LOGGER_H

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace ov_msckf {

/**
 * Fixed-capacity, best-effort logger for final estimator statistics.
 * Producers only copy into a bounded queue; a dedicated worker owns the output
 * descriptor and performs all writes.
 */
class BoundedStatsLogger {
public:
  static constexpr std::size_t kSlotCount = 16;
  static constexpr std::size_t kMaxRecordBytes = 4096;

  struct Snapshot {
    std::uint64_t enqueued = 0;
    std::uint64_t written = 0;
    std::uint64_t dropped_full = 0;
    std::uint64_t dropped_contended = 0;
    std::uint64_t dropped_sink = 0;
    std::uint64_t truncated = 0;
    std::uint64_t queued = 0;
  };

  explicit BoundedStatsLogger(int output_fd = STDOUT_FILENO) : output_fd_(output_fd) {
    worker_ = std::thread(&BoundedStatsLogger::worker_main, this);
  }

  BoundedStatsLogger(const BoundedStatsLogger &) = delete;
  BoundedStatsLogger &operator=(const BoundedStatsLogger &) = delete;

  ~BoundedStatsLogger() { stop(); }

  bool try_enqueue(const char *data, std::size_t size) noexcept {
    if (size == 0) {
      return true; // Empty records are accepted as no-ops and need no queue slot.
    }
    if (data == nullptr && size != 0) {
      dropped_sink_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (stopping_.load(std::memory_order_acquire) || sink_failed_.load(std::memory_order_acquire)) {
      dropped_sink_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    std::unique_lock<std::mutex> lock(queue_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      dropped_contended_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (stopping_.load(std::memory_order_relaxed) || sink_failed_.load(std::memory_order_relaxed)) {
      dropped_sink_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (queue_size_ == kSlotCount) {
      dropped_full_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    const std::size_t count = size < kMaxRecordBytes ? size : kMaxRecordBytes;
    Slot &slot = slots_[queue_tail_];
    if (count != 0) {
      std::memcpy(slot.bytes.data(), data, count);
    }
    slot.size = count;
    queue_tail_ = (queue_tail_ + 1) % kSlotCount;
    ++queue_size_;
    queued_.store(queue_size_, std::memory_order_relaxed);
    enqueued_.fetch_add(1, std::memory_order_relaxed);
    if (size > count) {
      truncated_.fetch_add(1, std::memory_order_relaxed);
    }
    lock.unlock();
    queue_cv_.notify_one();
    return true;
  }

  Snapshot counters() const noexcept {
    Snapshot result;
    result.enqueued = enqueued_.load(std::memory_order_relaxed);
    result.written = written_.load(std::memory_order_relaxed);
    result.dropped_full = dropped_full_.load(std::memory_order_relaxed);
    result.dropped_contended = dropped_contended_.load(std::memory_order_relaxed);
    result.dropped_sink = dropped_sink_.load(std::memory_order_relaxed);
    result.truncated = truncated_.load(std::memory_order_relaxed);
    result.queued = queued_.load(std::memory_order_relaxed);
    return result;
  }

  void stop() {
    std::lock_guard<std::mutex> stop_lock(stop_mutex_);
    if (!joined_) {
      stopping_.store(true, std::memory_order_release);
      {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        dropped_sink_.fetch_add(queue_size_, std::memory_order_relaxed);
        queue_head_ = queue_tail_ = queue_size_ = 0;
        queued_.store(0, std::memory_order_relaxed);
      }
      queue_cv_.notify_all();
      if (worker_.joinable()) {
        // A regular-file kernel write can itself stall indefinitely. Producer
        // calls remain isolated, but joining necessarily waits for that write.
        worker_.join();
      }
      joined_ = true;
    }
  }

private:
  struct Slot {
    std::array<char, kMaxRecordBytes> bytes{};
    std::size_t size = 0;
  };

  static constexpr int kWriteTimeoutMs = 100;

  enum class WriteResult { Success, Timeout, SinkError };

  WriteResult write_with_deadline(int fd, const char *data, std::size_t size, bool atomic_pipe_write) const {
    if (size == 0) return WriteResult::Success;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kWriteTimeoutMs);
    std::size_t offset = 0;
    do {
      if (stopping_.load(std::memory_order_acquire)) return WriteResult::Timeout;
      if (std::chrono::steady_clock::now() >= deadline) return WriteResult::Timeout;
      const ssize_t result = ::write(fd, data + offset, size - offset);
      if (result > 0) {
        offset += static_cast<std::size_t>(result);
        if (atomic_pipe_write && offset != size) {
          return WriteResult::SinkError;
        }
        continue;
      }
      if (result == 0) {
        return WriteResult::SinkError;
      }
      if (errno == EINTR) {
        continue;
      }
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        return WriteResult::SinkError;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return WriteResult::Timeout;
      }
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      int timeout = static_cast<int>(remaining.count());
      if (timeout < 1) timeout = 1;
      struct pollfd descriptor;
      descriptor.fd = fd;
      descriptor.events = POLLOUT;
      descriptor.revents = 0;
      const int poll_result = ::poll(&descriptor, 1, timeout);
      if (poll_result < 0 && errno != EINTR) {
        return WriteResult::SinkError;
      }
      if (poll_result == 0) {
        return WriteResult::Timeout;
      }
      if (poll_result > 0 && (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return WriteResult::SinkError;
      }
    } while (offset < size);
    return WriteResult::Success;
  }

  void account_queued_drop() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    dropped_sink_.fetch_add(queue_size_, std::memory_order_relaxed);
    queue_head_ = queue_tail_ = queue_size_ = 0;
    queued_.store(0, std::memory_order_relaxed);
  }

  void emit_drop_summary(int fd, bool is_pipe, std::size_t pipe_buf) {
    const std::uint64_t full = dropped_full_.load(std::memory_order_relaxed);
    const std::uint64_t contended = dropped_contended_.load(std::memory_order_relaxed);
    const std::uint64_t sink = dropped_sink_.load(std::memory_order_relaxed);
    const std::uint64_t truncated = truncated_.load(std::memory_order_relaxed);
    if (full == reported_full_ && contended == reported_contended_ && sink == reported_sink_ &&
        truncated == reported_truncated_) {
      return;
    }
    char summary[256];
    const int formatted = std::snprintf(summary, sizeof(summary),
        "[BoundedStatsLogger] dropped_full=%llu dropped_contended=%llu dropped_sink=%llu truncated=%llu\n",
        static_cast<unsigned long long>(full), static_cast<unsigned long long>(contended),
        static_cast<unsigned long long>(sink), static_cast<unsigned long long>(truncated));
    if (formatted <= 0) return;
    std::size_t length = static_cast<std::size_t>(formatted);
    if (length >= sizeof(summary)) length = sizeof(summary) - 1;
    if (is_pipe && length > pipe_buf) length = pipe_buf;
    if (write_with_deadline(fd, summary, length, is_pipe) == WriteResult::Success) {
      reported_full_ = full;
      reported_contended_ = contended;
      reported_sink_ = sink;
      reported_truncated_ = truncated;
    }
  }

  void worker_main() {
    sigset_t blocked_signals;
    ::sigemptyset(&blocked_signals);
    ::sigaddset(&blocked_signals, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &blocked_signals, nullptr);

    struct stat original_status;
    int sink_fd = -1;
    bool regular_file = false;
    if (::fstat(output_fd_, &original_status) == 0) {
      regular_file = S_ISREG(original_status.st_mode);
      if (regular_file) {
        // A regular file must share the original open-file-description offset:
        // reopening it with O_APPEND could make later caller writes overwrite
        // logger output when stdout was opened with shell redirection.
        sink_fd = ::fcntl(output_fd_, F_DUPFD_CLOEXEC, 0);
      } else {
        char path[64];
        const int path_size = std::snprintf(path, sizeof(path), "/proc/self/fd/%d", output_fd_);
        if (path_size > 0 && static_cast<std::size_t>(path_size) < sizeof(path)) {
          // Reopening pipes/FIFOs/terminals gives an independent description,
          // so nonblocking mode never changes the caller's original flags.
          sink_fd = ::open(path, O_WRONLY | O_NONBLOCK | O_APPEND | O_CLOEXEC);
        }
      }
    }
    if (sink_fd < 0) {
      sink_failed_.store(true, std::memory_order_release);
      account_queued_drop();
      return;
    }

    struct stat status;
    const bool is_pipe = (::fstat(sink_fd, &status) == 0 && S_ISFIFO(status.st_mode));
    long pipe_buf_long = is_pipe ? ::fpathconf(sink_fd, _PC_PIPE_BUF) : -1;
    std::size_t pipe_buf = pipe_buf_long > 0 ? static_cast<std::size_t>(pipe_buf_long) : 512;
    if (pipe_buf > kMaxRecordBytes) pipe_buf = kMaxRecordBytes;

    for (;;) {
      Slot record;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return stopping_.load(std::memory_order_acquire) || queue_size_ != 0; });
        if (stopping_.load(std::memory_order_acquire)) {
          dropped_sink_.fetch_add(queue_size_, std::memory_order_relaxed);
          queue_head_ = queue_tail_ = queue_size_ = 0;
          queued_.store(0, std::memory_order_relaxed);
          break;
        }
        record = slots_[queue_head_];
        queue_head_ = (queue_head_ + 1) % kSlotCount;
        --queue_size_;
        queued_.store(queue_size_, std::memory_order_relaxed);
      }

      std::size_t write_size = record.size;
      if (is_pipe && write_size > pipe_buf) {
        write_size = pipe_buf;
        truncated_.fetch_add(1, std::memory_order_relaxed);
      }
      const WriteResult write_result = write_with_deadline(sink_fd, record.bytes.data(), write_size, is_pipe);
      if (write_result != WriteResult::Success) {
        dropped_sink_.fetch_add(1, std::memory_order_relaxed);
        if (write_result == WriteResult::SinkError) {
          sink_failed_.store(true, std::memory_order_release);
          account_queued_drop();
          break;
        }
        // A timeout drops this record; a later successful record emits a summary.
        continue;
      }
      written_.fetch_add(1, std::memory_order_relaxed);
      emit_drop_summary(sink_fd, is_pipe, pipe_buf);
    }
    ::close(sink_fd);
  }

  const int output_fd_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::array<Slot, kSlotCount> slots_{};
  std::size_t queue_head_ = 0;
  std::size_t queue_tail_ = 0;
  std::size_t queue_size_ = 0;

  std::atomic<bool> stopping_{false};
  std::atomic<bool> sink_failed_{false};
  std::atomic<std::uint64_t> enqueued_{0};
  std::atomic<std::uint64_t> written_{0};
  std::atomic<std::uint64_t> dropped_full_{0};
  std::atomic<std::uint64_t> dropped_contended_{0};
  std::atomic<std::uint64_t> dropped_sink_{0};
  std::atomic<std::uint64_t> truncated_{0};
  std::atomic<std::uint64_t> queued_{0};

  std::thread worker_;
  std::mutex stop_mutex_;
  bool joined_ = false;
  std::uint64_t reported_full_ = 0;
  std::uint64_t reported_contended_ = 0;
  std::uint64_t reported_sink_ = 0;
  std::uint64_t reported_truncated_ = 0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_BOUNDED_STATS_LOGGER_H

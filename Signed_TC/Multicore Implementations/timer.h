#pragma once
#include <chrono>
#include <cstdio>
#include <string>

class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_time;

public:
  Timer() { reset(); }

  inline void reset() {
    start_time = std::chrono::high_resolution_clock::now();
  }

  inline double elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time).count();
  }

  inline double elapsed_sec() const { return elapsed_ms() / 1000.0; }

  inline void print(const char *label) const {
    std::printf("[%s] took %.4f seconds\n", label, elapsed_sec());
  }
};

class BlockTimer {
private:
  std::string label;
  Timer timer;

public:
  explicit BlockTimer(std::string block_label)
      : label(std::move(block_label)) {}

  ~BlockTimer() { timer.print(label.c_str()); }
};
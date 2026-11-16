// A small thread pool: a fixed set of worker threads taking jobs off one queue. wait() returns when every job handed
// in so far has finished. Built without threads (ROBOTNAV_NO_THREADS, as the WebAssembly build is), jobs simply run
// where they are handed in.
#pragma once

#include <cstddef>
#include <functional>
#include <utility>

#ifndef ROBOTNAV_NO_THREADS
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#endif

namespace robotnav {

#ifndef ROBOTNAV_NO_THREADS

class ThreadPool {
public:
    explicit ThreadPool(int threads) {
        for (int k = 0; k < (threads < 1 ? 1 : threads); ++k) workers_.emplace_back([this] { work(); });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            closing_ = true;
        }
        wake_.notify_all();
        for (std::thread& t : workers_) t.join();
    }

    int size() const { return static_cast<int>(workers_.size()); }

    void run(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(m_);
            jobs_.push_back(std::move(job));
            ++pending_;
        }
        wake_.notify_one();
    }

    void wait() {                                    // until the queue is empty and nothing is running
        std::unique_lock<std::mutex> lock(m_);
        idle_.wait(lock, [this] { return pending_ == 0; });
    }

    static int hardware() {                          // how many threads this machine runs at once (at least 1)
        const unsigned n = std::thread::hardware_concurrency();
        return n ? static_cast<int>(n) : 1;
    }

private:
    void work() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [this] { return closing_ || !jobs_.empty(); });
                if (jobs_.empty()) return;           // closing, and nothing left to do
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
            {
                std::lock_guard<std::mutex> lock(m_);
                --pending_;
            }
            idle_.notify_all();
        }
    }

    std::mutex m_;
    std::condition_variable wake_, idle_;
    std::deque<std::function<void()>> jobs_;
    std::vector<std::thread> workers_;
    std::size_t pending_ = 0;
    bool closing_ = false;
};

#else

class ThreadPool {
public:
    explicit ThreadPool(int) {}
    int size() const { return 1; }
    void run(std::function<void()> job) { job(); }
    void wait() {}
    static int hardware() { return 1; }
};

#endif

}  // namespace robotnav

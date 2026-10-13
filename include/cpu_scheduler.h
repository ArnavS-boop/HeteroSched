#ifndef CPU_SCHEDULER_H
#define CPU_SCHEDULER_H

#include "common.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>

class CPUScheduler {
public:
    explicit CPUScheduler(size_t num_workers);
    ~CPUScheduler();

    void Enqueue(TaskDescriptor* task);
    std::future<void> EnqueueWithFuture(TaskDescriptor* task);
    size_t PendingTasks();

private:
    void WorkerLoop(size_t worker_id);
    void ExecuteCPUTask(TaskDescriptor* task, size_t core_id);

    std::vector<std::thread> workers_;
    std::queue<std::pair<TaskDescriptor*, std::promise<void>*>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_requested_;
};

#endif

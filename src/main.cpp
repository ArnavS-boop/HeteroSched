#include "heterosched_api.h"
#include "workload_types.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdlib>

bool allocate_task_buffers(TaskDescriptor* task);
void free_task_buffers(TaskDescriptor* task);
void gpu_warmup();
void print_gpu_info();

struct TraceEntry {
    uint64_t id;
    std::string type;
    uint32_t priority;
    size_t dim_m, dim_n, dim_k;
    size_t bytes_in, bytes_out;
    uint64_t estimated_flops;
};

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n\"");
    size_t end = s.find_last_not_of(" \t\r\n\"");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static std::vector<TraceEntry> load_trace(const std::string& path) {
    std::vector<TraceEntry> entries;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open trace file: " << path << "\n";
        return entries;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    TraceEntry current;
    bool in_object = false;

    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line == "[" || line == "]") continue;

        if (line == "{") {
            in_object = true;
            current = TraceEntry{};
            continue;
        }

        if (line == "}" || line == "},") {
            if (in_object) {
                entries.push_back(current);
                in_object = false;
            }
            continue;
        }

        if (!in_object) continue;

        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;

        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (!val.empty() && val.back() == ',') val.pop_back();
        val = trim(val);

        if (key == "id")              current.id = std::stoull(val);
        else if (key == "type")       current.type = val;
        else if (key == "priority")   current.priority = std::stoul(val);
        else if (key == "dim_m")      current.dim_m = std::stoull(val);
        else if (key == "dim_n")      current.dim_n = std::stoull(val);
        else if (key == "dim_k")      current.dim_k = std::stoull(val);
        else if (key == "bytes_in")   current.bytes_in = std::stoull(val);
        else if (key == "bytes_out")  current.bytes_out = std::stoull(val);
        else if (key == "estimated_flops") current.estimated_flops = std::stoull(val);
    }

    return entries;
}

static void write_results_csv(const std::string& path,
                               const std::vector<TaskDescriptor>& tasks,
                               double total_time_ms) {
    std::ofstream out(path);
    out << "task_id,type,priority,dim_m,dim_n,dim_k,"
        << "device,t_decision_us,t_transfer_us,t_exec_us,"
        << "t_sync_us,t_total_latency_us,stream_id,cpu_core\n";

    for (auto& t : tasks) {
        out << t.task_id << ","
            << TaskTypeToString(t.type) << ","
            << t.priority << ","
            << t.dim_m << "," << t.dim_n << "," << t.dim_k << ","
            << DeviceToString(t.metrics.assigned_device) << ","
            << t.metrics.t_decision_us << ","
            << t.metrics.t_transfer_us << ","
            << t.metrics.t_exec_us << ","
            << t.metrics.t_sync_us << ","
            << t.metrics.t_total_latency_us << ","
            << t.metrics.stream_id_used << ","
            << t.metrics.cpu_core_used << "\n";
    }

    out << "\n# total_time_ms," << total_time_ms << "\n";
    out.close();
}

static void print_summary(const std::vector<TaskDescriptor>& tasks, double total_time_ms) {
    size_t cpu_count = 0, gpu_count = 0;
    double total_decision = 0, total_exec = 0, total_transfer = 0;

    for (auto& t : tasks) {
        if (t.metrics.assigned_device == ExecutionDevice::DEVICE_CPU) cpu_count++;
        else gpu_count++;
        total_decision += t.metrics.t_decision_us;
        total_exec += t.metrics.t_exec_us;
        total_transfer += t.metrics.t_transfer_us;
    }

    std::cout << "\n=== HeteroSched Results ===\n"
              << "Total tasks: " << tasks.size() << "\n"
              << "  CPU: " << cpu_count << "  GPU: " << gpu_count << "\n"
              << "Total wall time: " << total_time_ms << " ms\n"
              << "Throughput: " << (tasks.size() / (total_time_ms / 1000.0)) << " tasks/sec\n"
              << "Avg decision overhead: " << (total_decision / tasks.size()) << " us\n"
              << "Avg exec time: " << (total_exec / tasks.size()) << " us\n"
              << "Avg transfer time (GPU tasks): "
              << (gpu_count > 0 ? total_transfer / gpu_count : 0) << " us\n"
              << "===========================\n";
}

int main(int argc, char** argv) {
    std::string trace_path = "benchmarks/traces/mixed_trace.json";
    std::string output_path = "results.csv";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--trace" && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    print_gpu_info();
    gpu_warmup();

    auto trace_entries = load_trace(trace_path);
    if (trace_entries.empty()) {
        std::cerr << "No tasks loaded from trace. Exiting.\n";
        return 1;
    }

    std::cout << "Loaded " << trace_entries.size() << " tasks from " << trace_path << "\n";

    std::vector<TaskDescriptor> tasks(trace_entries.size());
    for (size_t i = 0; i < trace_entries.size(); ++i) {
        auto& e = trace_entries[i];
        auto& t = tasks[i];
        memset(&t, 0, sizeof(TaskDescriptor));
        t.task_id = e.id;
        t.type = StringToTaskType(e.type);
        t.priority = e.priority;
        t.dim_m = e.dim_m;
        t.dim_n = e.dim_n;
        t.dim_k = e.dim_k;
        t.bytes_in = e.bytes_in;
        t.bytes_out = e.bytes_out;
        t.estimated_flops = e.estimated_flops;
        bool alloc_ok = allocate_task_buffers(&t);
        if (!alloc_ok) {
            std::cerr << "Warning: Failed to allocate buffers for task ID " << t.task_id << std::endl;
        }
    }

    HeteroSchedConfig config;
    config.cpu_workers = 4;
    config.gpu_stream_pool = 4;
    config.enable_tier1 = true;
    config.enable_tier2 = true;
    config.enable_dynamic_penalties = true;
    config.enable_cross_layer_priority = true;

    HeteroSchedRuntime runtime(config);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (auto& task : tasks) {
        auto task_start = std::chrono::high_resolution_clock::now();
        runtime.SubmitTask(&task);
        auto task_end = std::chrono::high_resolution_clock::now();
        task.metrics.t_total_latency_us =
            std::chrono::duration<double, std::micro>(task_end - task_start).count();
    }

    runtime.WaitAll();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    print_summary(tasks, total_time_ms);
    write_results_csv(output_path, tasks, total_time_ms);
    std::cout << "Results written to " << output_path << "\n";

    for (auto& t : tasks) {
        free_task_buffers(&t);
    }

    return 0;
}

HeteroSchedRuntime::HeteroSchedRuntime(const HeteroSchedConfig& config)
    : config_(config),
      cpu_scheduler_(config.cpu_workers),
      gpu_executor_(config.gpu_stream_pool),
      controller_(std::make_unique<DecisionForestModel>()),
      gpu_round_robin_(0) {
    controller_.SetTier1Enabled(config.enable_tier1);
    controller_.SetTier2Enabled(config.enable_tier2);
    controller_.SetDynamicPenaltiesEnabled(config.enable_dynamic_penalties);
}

HeteroSchedRuntime::~HeteroSchedRuntime() {
    WaitAll();
}

void HeteroSchedRuntime::SubmitTask(TaskDescriptor* task) {
    if (!task) return;
    if (!task->host_input || !task->host_output) {
        task->metrics.success = false;
        return;
    }
    RuntimeState state = telemetry_.Snapshot();
    ExecutionDevice device = controller_.ScheduleTask(task, state);

    if (device == ExecutionDevice::DEVICE_GPU) {
        ApplyCrossLayerPriority(task, device);
        telemetry_.RecordTransferStart(task->bytes_in + task->bytes_out);
        telemetry_.IncrementGPUQueue();

        size_t stream_idx;
        {
            std::lock_guard<std::mutex> lock(submit_mutex_);
            stream_idx = gpu_round_robin_++;
        }

        gpu_executor_.ExecuteAsync(task, stream_idx);

        telemetry_.DecrementGPUQueue();
        telemetry_.RecordTransferEnd(task->bytes_in + task->bytes_out);
    } else {
        telemetry_.IncrementCPUQueue();
        auto future = cpu_scheduler_.EnqueueWithFuture(task);

        {
            std::lock_guard<std::mutex> lock(submit_mutex_);
            pending_futures_.push_back(std::move(future));
        }

        telemetry_.DecrementCPUQueue();
    }
}

void HeteroSchedRuntime::WaitAll() {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    for (auto& f : pending_futures_) {
        if (f.valid()) f.wait();
    }
    pending_futures_.clear();
}

void HeteroSchedRuntime::ApplyCrossLayerPriority(TaskDescriptor* task, ExecutionDevice device) {
    if (!config_.enable_cross_layer_priority) return;
    if (device != ExecutionDevice::DEVICE_GPU) return;

    if (task->priority >= 2) {
        struct sched_param param;
        param.sched_priority = 1;
        pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    }
}

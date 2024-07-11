#ifndef MINDSPORE_OFFLOADING_SCHEDULER_H
#define MINDSPORE_OFFLOADING_SCHEDULER_H

#include "utils.h"
#include "task_queue.h"
#include "stage_executor.h"

using offloading_serving::PredictRequest;
using offloading_serving::PredictReply;

namespace mindspore {

namespace offloading {

class StageExecutor;
class PreprocessExecutor;
class InferenceExecutor;

struct InferenceSession {
  PredictOnFinish callback;
  PredictReply* reply;

  InferenceSession(PredictOnFinish cb, PredictReply* r) : callback(cb), reply(r) {}
};
using InferenceSessionPtr = std::shared_ptr<InferenceSession>;

void SeparateBatchTensor(TensorPtr& tensor, std::vector<offloading_serving::PredictReply*>& replies);

class Scheduler : public std::enable_shared_from_this<Scheduler> {
  public:
    Scheduler(ExecutionMode exec_mode, int64_t max_wait_time_ms, size_t max_batch_size, size_t load_window_size) 
    : mode_(exec_mode), 
      payload_queue_(max_wait_time_ms, max_batch_size),
      k_buf_(load_window_size),
      q_time_buf_(load_window_size) {}
    ~Scheduler() = default;
    void Init(OffloadingServerConfig &cfg, const std::string &path);
    void Stop();
    void ProcessRequest(offloading_serving::PredictRequest* request,
                        offloading_serving::PredictReply* response,
                        PredictOnFinish& callback);
    void EnqueueCompletionQueue(TaskPtr&& task);
    void GetRuntimeProfile(double &factor, double &q_time);
  private:
    void Run();
    void ScheduleNoBatching();
    void ScheduleNaiveBatching();
    void ScheduleSNBBatching();
    void ProcessResponse();
    uint64_t GetNextID();
    // executor
    ExecutionMode mode_;
    std::vector<std::shared_ptr<StageExecutor>> stages_;
    PayloadQueue payload_queue_;
    TaskQueue completion_queue_;
    std::thread sched_thread_;
    std::thread completion_thread_;
    std::atomic<bool> is_sched_running_ = false;
    std::atomic<bool> is_completion_running_ = false;
    std::mutex inf_session_map_lock_;
    std::map<uint64_t, InferenceSessionPtr> inf_session_map_;

    // for measurement
    FixedSizeBuffer k_buf_;
    FixedSizeBuffer q_time_buf_;
    std::mutex mtx_;

    // for time log
    std::ofstream time_log_ofs;
};

}
}

#endif
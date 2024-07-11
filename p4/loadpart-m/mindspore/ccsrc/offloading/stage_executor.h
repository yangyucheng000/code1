#ifndef MINDSPORE_OFFLOADING_EXECUTOR_H
#define MINDSPORE_OFFLOADING_EXECUTOR_H

#include "utils.h"
#include "cost_graph.h"
#include "task_queue.h"
#include "payload.h"
#include "scheduler.h"
#include "quant.h"

namespace mindspore {

namespace offloading {

class Scheduler;

class StageExecutor {
  public:
    StageExecutor() = default;
    ~StageExecutor() = default;
    void Init();
    virtual void Stop();
    void SetNext(const std::shared_ptr<StageExecutor>& next) { next_executor_ = next; }
    void SetScheduler(const std::shared_ptr<Scheduler>& sched) { sched_ = sched; }
    void SetLastStage(bool is_last_stage) { is_last_stage_ = is_last_stage; }
  protected:
    virtual void Run();
    std::thread executor_;
    std::atomic<bool> is_running_;
    std::weak_ptr<Scheduler> sched_;
    std::shared_ptr<StageExecutor> next_executor_ = nullptr;
    bool is_last_stage_ = false;
};

class PreprocessExecutor : public StageExecutor {
  public:
    PreprocessExecutor(int decompress_nthreads, bool is_dequant)
      : StageExecutor(), queue_(0, 1), decompressor_(decompress_nthreads), is_dequant_(is_dequant) {}
    void Run();
    void EnqueuePayload(PayloadPtr& payload);
  private:
    void ConstructTaskInputs(PayloadPtr& payload, std::vector<NameToQTensorMap>& inputs);

    PayloadQueue queue_;
    // Decompressors
    Decompressor decompressor_;
    bool is_dequant_ = false;
};

class InferenceExecutor : public StageExecutor {
  public:
    InferenceExecutor(bool is_dequant) : StageExecutor(), is_dequant_(is_dequant) {}
    void Init(const std::string &path, const std::string &dequant_path, size_t max_batch_size, bool prep_full_graphs);
    void Run();
    void Stop();
    void EnqueueTask(TaskPtr&& task);
  private: 
    Status InitEnv();
    Status FinalizeEnv();
    Status CompileGraph(const FuncGraphPtr &func_graph, GraphId &graph_id);
    TaskPtr ExecuteTaskNoBatching(TaskPtr&& task);
    TaskPtr ExecuteTaskNaiveBatching(TaskPtr&& task);
    TaskPtr ExecuteTaskSNBBatching(TaskPtr&& task);
    // map of <KernelGraphPtr, GraphId>
    TaskQueue queue_;

    bool is_env_init_ = false;
    BatchOffloadingContextCache context_cache_;
    session::SessionPtr session_impl_ = nullptr;
    std::shared_ptr<CostGraph> cost_graph_;
    LatencyGraphManager latency_graph_manager_;
    // Dequantizers
    bool is_dequant_ = false;
    std::unique_ptr<Dequantizer> dequantizer_ = nullptr;
};

tensor::TensorPtr MergeTensorPair(tensor::TensorPtr &tensor_a, tensor::TensorPtr &tensor_b);

void LoadIntermidiateData(tensor::TensorPtr &tensor);

}
}

#endif
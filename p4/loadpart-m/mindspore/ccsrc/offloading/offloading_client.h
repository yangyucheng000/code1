#ifndef MINDSPORE_OFFLOADING_CLIENT_H
#define MINDSPORE_OFFLOADING_CLIENT_H

#include <grpcpp/grpcpp.h>
#include <vector>
#include <string>
#include <fstream>
#include "utils.h"
#include "status.h"
#include "quant.h"
#include "cost_graph.h"

using offloading_serving::BatchOffloadingService;
using offloading_serving::PredictRequest;
using offloading_serving::PredictReply;
using offloading_serving::DeployRequest;
using offloading_serving::DeployReply;
using offloading_serving::ProfileRequest;
using offloading_serving::ProfileReply;
using offloading_serving::SimpleRequest;
using offloading_serving::SimpleReply;
using offloading_serving::GraphProfile;

namespace mindspore {

namespace offloading {
  
namespace py = pybind11;

class OffloadingClient {
  public:
    struct AsyncPredictCall {
      PredictReply reply;
      grpc::ClientContext context;
      grpc::Status status;
      std::unique_ptr<grpc::ClientAsyncResponseReader<PredictReply>> response_reader;
    };

    explicit OffloadingClient(std::shared_ptr<grpc::Channel> channel, OffloadingClientConfig config) 
      : stub_(offloading_serving::BatchOffloadingService::NewStub(channel)), 
        cfg_(config),
        compressor_(Compressor(cfg_.compress_clevel, cfg_.compress_do_shuffle, cfg_.compress_nthreads, cfg_.compressor)),
        up_buf_(cfg_.bandwidth_window_size),
        factor_buf_(cfg_.load_window_size),
        q_time_buf_(cfg_.load_window_size) {}
    ~OffloadingClient();
    void Stop();
    void Clear();
    py::object Predict(const std::unordered_map<std::string, py::object> &kwargs);
    py::object PredictOracle(const std::unordered_map<std::string, py::object> &kwargs, double bdw, double q_time, double load_factor);
    py::object StaticPredictTest(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    int PrintCNodeOrder();
    void Deploy(const std::string &path);
    void Profile(const std::string &path);
    // for experiment
    void PartitionDecisionTest(double bdw, double q_time, double load_factor);
    void FakeDeploy(const std::string &path);
    void FakeStaticPredictTest(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    py::object PartialExecute(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    py::object ConvertProtoTensorFromFile(const std::string &path);
    // for test
    py::object UnitTest(const std::unordered_map<std::string, py::object> &kwargs);
  private:
    py::object PredictInner(const size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes, const std::unordered_map<std::string, py::object> &kwargs);
    bool ProcessArg(const std::unordered_map<std::string, py::object> &kwargs, OffloadingContext& ctx, const bool is_full_offloading);
    bool ProfileGraph(std::unordered_map<std::string, float> &result);
    bool ProfileGraph(KernelGraphPtr &kg, std::unordered_map<std::string, float> &result);
    void MeasureBandWidth();
    void MeasureLoad();
    Status InitEnv();
    Status FinalizeEnv();
    Status CompileGraph(const FuncGraphPtr &func_graph, GraphId &graph_id);
    void ConvertProfileResToCSV(const std::string &path);
    // for experiment
    void AsyncCompleteFakePredict();

    bool is_init_ = false;
    bool is_deployed_ = false;

    std::unique_ptr<BatchOffloadingService::Stub> stub_;
    OffloadingClientConfig cfg_;

    OffloadingContextCache context_cache_;
    session::SessionPtr session_impl_ = nullptr;
    std::shared_ptr<CostGraph> cost_graph_;
    LatencyGraphManager latency_graph_manager_;

    Compressor compressor_;
    std::unique_ptr<Quantizer> quantizer_;

    // for measure
    std::thread measure_thread_;
    std::mutex mtx_;
    std::mutex measure_mtx_;
    FixedSizeBuffer up_buf_;
    FixedSizeBuffer factor_buf_;
    FixedSizeBuffer q_time_buf_;
    std::atomic<bool> is_measure_running_ = false;

    // for experiment
    std::ofstream log_file;
    grpc::CompletionQueue cq_;
    bool is_deployed_fake_ = false;
    std::thread fake_pred_completion_thread_;
};

}
}
#endif
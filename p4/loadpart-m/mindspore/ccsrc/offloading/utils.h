#ifndef MINDSPORE_OFFLOADING_UTILS_H
#define MINDSPORE_OFFLOADING_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <dbg.h>
#include "proto/offloading_service.grpc.pb.h"
#include "ir/tensor.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "utils/convert_utils.h"
#include "utils/convert_utils_py.h"
#include "ir/param_info.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "base/base_ref_utils.h"
#include "vm/segment_runner.h"
#include "debug/draw.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/executor_manager.h"
#include "backend/session/session_basic.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/session_factory.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "vm/backend.h"
#include "vm/transform.h"
#include "pipeline/jit/base.h"
#include "pipeline/jit/parse/data_converter.h"
#include "load_mindir/load_model.h"
#include "profiler/device/cpu/cpu_profiling.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#if ENABLE_GPU
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "profiler/device/gpu/gpu_profiling.h"
#include "runtime/device/gpu/cuda_driver.h"
#endif

namespace mindspore {

namespace offloading {

#define TIMESTAMP() std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
#define TIMESTAMP_DIFF_MICROSECONDS(start, end) std::chrono::duration_cast<std::chrono::microseconds>((end) - (start)).count()

namespace py = pybind11;

using namespace std::chrono_literals;

using KernelGraph = mindspore::session::KernelGraph;

using UsedInputMap = std::unordered_map<std::string, std::pair<tensor::TensorPtr, bool>>;
using NameToTensorMap = std::unordered_map<std::string, tensor::TensorPtr>;
using PrimitiveSet = std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual>;

using PredictOnFinish = std::function<void (offloading_serving::PredictReply*, grpc::Status)>;

const uint32_t kInvalidGraphId = UINT32_MAX;

using NameToQTensorMap = std::unordered_map<std::string, std::vector<tensor::TensorPtr>>;

struct OffloadingServerConfig {
  int max_msg_mb_size;
  uint64_t measure_interval;
  int profile_times;
  int decompress_nthreads;
  int64_t max_wait_time_ms;
  size_t max_batch_size;
  int exec_mode;
  std::string dequant_path;
  bool is_decompress;
  bool is_dequant;
  bool prep_full_graphs;
  bool time_log;
  size_t load_window_size;
};

struct OffloadingClientConfig {
  int max_msg_mb_size;
  size_t bandwidth_window_size;
  size_t load_window_size;
  int64_t measure_interval;
  int profile_times;
  bool is_static;
  bool load_control; 
  bool passive_measure;
  int compress_clevel;
  int compress_do_shuffle;
  int compress_nthreads;
  std::string compressor;
  std::string quant_path_prefix;
  bool is_compress;
  bool is_quant;
  bool local_deploy;
  double profile_scale_factor;
  double oracle_bdw; // in MB/s
};

struct OffloadingContext {
  GraphId graph_id;
  KernelGraphPtr graph;
  tensor::TensorPtrList input_tensors; // set when first inference
  std::unordered_map<std::string, size_t> input_name_list_idx_map; // set when first inference
  std::vector<std::string> output_names; // set when partition
  OffloadingContext() : graph_id(kInvalidGraphId), graph(nullptr) {}
  OffloadingContext(const GraphId gid, KernelGraphPtr g) : graph_id(gid), graph(g) {}
};

struct OffloadingContextCache {
  // cut_node -> context
  std::unordered_map<std::string, OffloadingContext> cache_;
  std::unordered_map<std::string, size_t> cut_nodes_to_lg_idx_;
  FuncGraphPtr full_func_graph_;
  FuncGraphManagerPtr full_graph_manager_;

  void SetFuncGraph(FuncGraphPtr g) { full_func_graph_ = g; }
  void SetFullGraphManager(FuncGraphManagerPtr manager) { full_graph_manager_ = manager; }

  bool FindContext(const std::string &cut_nodes_string);
  void AddContext(const size_t lg_idx, const std::string &cut_nodes_string, const GraphId gid, KernelGraphPtr g, std::vector<std::string> &output_names);
  OffloadingContext& GetContext(const std::string &cut_nodes_string);
};

struct BatchOffloadingContext {
  // we assume all graphs with different batch sizes share the same set of Parameters
  std::unordered_map<size_t, std::pair<GraphId, KernelGraphPtr>> graphs;
  tensor::TensorPtrList input_tensors; // set when first inference
  std::unordered_map<std::string, size_t> input_name_list_idx_map; // set when first inference
  std::vector<std::string> output_names; // set when partition
  double base_time;

  BatchOffloadingContext() = default;
  BatchOffloadingContext(size_t batch_size, const GraphId gid, KernelGraphPtr g);

  bool FindGraph(size_t batch_size);
  void AddGraph(size_t batch_size, const GraphId gid, KernelGraphPtr g, std::vector<std::string> &o_names);
  std::pair<GraphId, KernelGraphPtr>& GetGraph(size_t batch_size);
  void SetBaseTime(double time) { base_time = time; }
  double GetBaseTime() { return base_time; }

  void GenerateInputVectors(KernelGraphPtr &graph);
};

struct BatchOffloadingContextCache {
  // start_cut_node -> end_cut_node -> context
  std::unordered_map<std::string, std::unordered_map<std::string, BatchOffloadingContext>> cache_;
  std::unordered_map<std::string, size_t> cut_nodes_to_lg_idx_;
  FuncGraphPtr full_func_graph_;
  std::unordered_map<size_t, FuncGraphManagerPtr> full_graph_managers_;

  void SetFuncGraph(FuncGraphPtr g) { full_func_graph_ = g; }
  void SetFullGraphManager(size_t batch_size, FuncGraphManagerPtr manager) { full_graph_managers_[batch_size] = manager; }
  
  bool FindContext(const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string);
  BatchOffloadingContext& AddContext(const size_t start_lg_idx, const size_t end_lg_idx, const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string);
  BatchOffloadingContext& GetContext(const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string);

  std::pair<GraphId, KernelGraphPtr> GenerateFullKernelGraphByBatchSize(session::SessionPtr session_impl, size_t batch_size);
};

enum ExecutionMode {
  NO_BATCHING,
  NAIVE_BATCHING,
  SNB_BATCHING
};

struct ExecutionPlan {
  // len(entry_list) == len(bsz_list)
  std::vector<std::pair<size_t, std::string>> entry_list;
  std::vector<size_t> bsz_list;
  ExecutionMode mode;
};

class FixedSizeBuffer {
  private:
    std::deque<double> buf_;
    size_t max_size_;
  public:
    FixedSizeBuffer(const size_t max_size) : max_size_(max_size) {}
    void PopFront();
    void PopBack();
    void Push(const double x);
    double GetAvgValue();
    bool IsEmpty();
    void Clear();
};

template<typename T>
void PrintVector(std::vector<T> &v, std::ofstream &ofs) {
  if (v.empty()) {
    ofs << "[]";
    return;
  }
  ofs << "[";
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ofs << v[i] << ", ";
  }
  ofs << v.back() << "]";
}

TypeId GetTypeIdFromProtoTensor(const offloading_serving::TensorProto &tensor_proto);

bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set);

void UpdateKernelArgs(CNodePtr &cnode);

bool CheckAllTensor(const ValueTuplePtr &value_tuple);

AbstractBasePtr ValueToAbstract(const ValuePtr &value, bool enable_tuple_broaden = false);

offloading_serving::TensorProto_DataType GetProtoDataType(TypeId type_id);

void TensorToProtoTensor(const tensor::TensorPtr &tensor, offloading_serving::TensorProto *const tensor_proto);

tensor::TensorPtr ProtoTensorToTensor(const offloading_serving::TensorProto &tensor_proto);

py::object TensorToPyData(const tensor::TensorPtr &tensor);

tensor::TensorPtr PyDataToTensor(const py::object &obj);

bool CheckInputArgShape(const ValuePtr &value, const ParameterPtr &input_node);

void GetOrderedCnodesVector(FuncGraphPtr &graph, CNodePtrList &cnodes);

bool CompareInput(const tensor::TensorPtr &input, const ParameterPtr &parameter);

void PrintParameterShape(const ParameterPtr &parameter);

void GetSegmentOutput(KernelGraphPtr &origin_graph, const CNodePtrList &node_list, AnfNodePtrList &output_nodes);

void PrintTensorData(tensor::TensorPtr &tensor);

AnfNodePtr TranslateAnfNode(KernelGraphPtr &from_g, KernelGraphPtr &to_g, const AnfNodePtr &node);

void TranslateCNodeList(KernelGraphPtr &from_g, KernelGraphPtr &to_g, CNodePtrList &cnode_list);
}
}

#endif
#include "offloading_client.h"

namespace mindspore {

namespace offloading {

OffloadingClient::~OffloadingClient() {
  Stop();
  Clear();
}

void OffloadingClient::Stop() {
  if (is_deployed_fake_) {
    cq_.Shutdown();
    fake_pred_completion_thread_.join();
  }

  if (!cfg_.is_static && measure_thread_.joinable() && measure_thread_.get_id() != std::this_thread::get_id()) {
    if (is_measure_running_.load()) {
      is_measure_running_.store(false);
    }
    measure_thread_.join();
  }
}

void OffloadingClient::Clear() {
  if (FinalizeEnv() != SUCCESS) {
    MS_LOG(ERROR) << "Clear env failed!";
  }
  log_file.close();
}

py::object OffloadingClient::Predict(const std::unordered_map<std::string, py::object> &kwargs) {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Client: model is not loaded, please call Deploy";
    return py::none();
  }

  double bdw = 0.0, q_time = 0.0, factor = 1.0;
  if (!cfg_.is_static) {
    if (cfg_.load_control) {
      mtx_.lock();
      q_time = q_time_buf_.GetAvgValue();
      factor = factor_buf_.GetAvgValue();
      mtx_.unlock();
      bdw = cfg_.oracle_bdw;
    } else {
      mtx_.lock();
      bdw = up_buf_.GetAvgValue();
      mtx_.unlock();
    }
  } else {
    MS_LOG(EXCEPTION) << "Client: client in static mode, please call static_predict_test for offloading";
  }

  LatencyGraph::PartitionResult res;
  auto start_time = TIMESTAMP();
  auto lg_idx = latency_graph_manager_.PartitionDecision(bdw, q_time, factor, res);
  auto end_time = TIMESTAMP();
  auto pda_time = end_time - start_time;
  log_file << "PDA\t" << pda_time << std::endl;

  start_time = TIMESTAMP();
  auto ret = PredictInner(lg_idx, res.best_cut_nodes_, kwargs);
  end_time = TIMESTAMP();
  log_file << "\t" << end_time - start_time + pda_time << std::endl;
  return ret;
}

py::object OffloadingClient::PredictOracle(const std::unordered_map<std::string, py::object> &kwargs, double bdw, double q_time, double load_factor) {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Client: model is not loaded, please call Deploy";
    return py::none();
  }

  LatencyGraph::PartitionResult res;
  auto start_time = TIMESTAMP();
  auto lg_idx = latency_graph_manager_.PartitionDecision(bdw, q_time, load_factor, res);
  auto end_time = TIMESTAMP();
  auto pda_time = end_time - start_time;
  log_file << "PDA\t" << pda_time << std::endl;

  start_time = TIMESTAMP();
  auto ret = PredictInner(lg_idx, res.best_cut_nodes_, kwargs);
  end_time = TIMESTAMP();
  log_file << "\t" << end_time - start_time + pda_time << std::endl;
  return ret;
}

py::object OffloadingClient::StaticPredictTest(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Client: model is not loaded, please call Deploy";
    return py::none();
  }
  // find cut nodes by name
  std::unordered_set<CostGraph::NodePtr> cut_nodes;
  cost_graph_->GetNodesByName(cut_nodes_names, cut_nodes);
  auto start_time = TIMESTAMP();
  auto ret = PredictInner(lg_idx, cut_nodes, kwargs);
  auto end_time = TIMESTAMP();
  log_file << "\t" << end_time - start_time << std::endl;
  return ret;
}

void OffloadingClient::Deploy(const std::string &path) {
  if (is_init_ && is_deployed_) return;
  // Init env
  if (InitEnv() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Client: InitEnv failed";
    return;
  }
  // Load full model as FuncGraph
  MindIRLoader model_loader;
  FuncGraphPtr full_func_graph = model_loader.LoadMindIR(path);
  if (full_func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Client: load MindIR model failed";
    return;
  }
  GraphId full_graph_id;
  KernelGraphPtr full_graph;
  Status ret = CompileGraph(full_func_graph, full_graph_id);
  if (ret != SUCCESS) {
    MS_LOG(EXCEPTION) << "Client: compile full graph failed";
    return;
  }

  full_graph = session_impl_->GetGraph(full_graph_id);
  if (full_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Client: GetGraph from session failed";
    return;
  }
  FuncGraphManagerPtr manager = MakeManager({full_graph});
  if (manager) {
    manager->AddFuncGraph(full_graph);
    full_graph->set_manager(manager);
  }

  context_cache_.SetFuncGraph(full_func_graph);
  std::vector<std::string> output_name = {"return"};
  context_cache_.AddContext(0, "", full_graph_id, full_graph, output_name);
  context_cache_.SetFullGraphManager(manager);

  // load profile res from proto file and construct cost_graph
  cost_graph_ = std::make_shared<CostGraph>(full_graph, path.substr(0, path.find_first_of('.')) + "_local.prof", cfg_.profile_scale_factor);
  // load cloud time map from file
  GraphProfile cloud_time_map_proto;
  std::ifstream ifs(path.substr(0, path.find_first_of('.')) + "_remote.time", std::ios::binary);
  if (!cloud_time_map_proto.ParseFromIstream(&ifs)) {
    MS_LOG(EXCEPTION) << "Client: failed to load cloud time map proto file at " << path.substr(0, path.find_first_of('.')) + "_remote.time";
  }
  ifs.close();
  cost_graph_->SetCloudTimeMap(cloud_time_map_proto);
  latency_graph_manager_.SetCostGraph(cost_graph_);
  if (cfg_.is_quant && cfg_.is_compress) {
    latency_graph_manager_.LoadCpsProfile(path.substr(0, path.find_first_of('.')) + "_cps.time");
  }
  latency_graph_manager_.SplitCostGraphIntoLatencyGraphs(cfg_.is_quant, cfg_.is_compress);

  cost_graph_->DrawCostGraph(path.substr(0, path.find_first_of('.')) + "_cg.dot", true, true);
  // init quantizer
  quantizer_ = std::unique_ptr<Quantizer>(new Quantizer(cfg_.quant_path_prefix, session_impl_));

  log_file.open("client.log");
  if (!log_file.is_open()) {
    MS_LOG(EXCEPTION) << "Client: open file client.log failed!";
  }

  if (!cfg_.local_deploy) {
    DeployRequest request;
    request.set_file_name(path);
    // Container for the reply
    DeployReply reply;
    // Client context
    grpc::ClientContext context;
    // Do RPC
    grpc::Status status = stub_->Deploy(&context, request, &reply);
    // Act upon reply
    if (!status.ok()) {
      std::cout << "FAILED: " << status.error_message() << std::endl;
      return;
    }

    // Measurement heat up
    if (!cfg_.is_static) {
      if (!is_measure_running_.load()) {
        is_measure_running_.store(true);
      }
      if (cfg_.load_control) {
        measure_thread_ = std::thread(&OffloadingClient::MeasureLoad, this);
        std::cout << "Client: waiting for the first load test..." << std::endl;
        while (factor_buf_.IsEmpty()) { std::this_thread::sleep_for(100ms); }
      } else {
        measure_thread_ = std::thread(&OffloadingClient::MeasureBandWidth, this);
        std::cout << "Client: waiting for the first bandwidth test..." << std::endl;
        while (up_buf_.IsEmpty()) { std::this_thread::sleep_for(100ms); }
      }
    }
  }

  is_deployed_ = true;
}

void OffloadingClient::Profile(const std::string &path) {
  // Init env
  if (InitEnv() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Client: InitEnv failed";
    return;
  }
  // Load full model as FuncGraph
  MindIRLoader model_loader;
  FuncGraphPtr full_func_graph = model_loader.LoadMindIR(path);
  if (full_func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Client: load MindIR model failed";
    return;
  }
  GraphId full_graph_id;
  KernelGraphPtr full_graph;
  Status ret = CompileGraph(full_func_graph, full_graph_id);
  if (ret != SUCCESS) {
    MS_LOG(EXCEPTION) << "Client: compile full graph failed";
    return;
  }

  full_graph = session_impl_->GetGraph(full_graph_id);
  if (full_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Client: GetGraph from session failed";
    return;
  }
  FuncGraphManagerPtr manager = MakeManager({full_graph});
  if (manager) {
    manager->AddFuncGraph(full_graph);
    full_graph->set_manager(manager);
  }
  std::vector<std::string> output_name = {"return"};
  context_cache_.AddContext(0, "", full_graph_id, full_graph, output_name);
  // Profile KernelGraph and construct CostGraph
  std::unordered_map<std::string, float> profile_res;
  if (!ProfileGraph(profile_res)) {
    MS_LOG(EXCEPTION) << "Client: profile KernelGraph failed";
    return;
  }
  // construct proto and write to file
  GraphProfile profile_proto;
  GraphProfile::BatchProfileEntry* batch_profile_entry = profile_proto.add_entries();
  batch_profile_entry->set_batch_size(1);
  for (const auto& entry : profile_res) {
    GraphProfile::ProfileEntry* profile_entry = batch_profile_entry->add_profile();
    profile_entry->set_name(entry.first);
    profile_entry->set_time(entry.second);
  }
  std::ofstream ofs(path.substr(0, path.find_first_of('.')) + "_local.prof", std::ios::binary);
  if (!profile_proto.SerializeToOstream(&ofs)) {
    MS_LOG(EXCEPTION) << "Client: failed to serialize local GraphProfile";
  }
  ofs.close();

  ProfileRequest request;
  request.set_file_name(path);
  // Container for the reply
  ProfileReply reply;
  // Client context
  grpc::ClientContext context;
  // Do RPC
  grpc::Status status = stub_->Profile(&context, request, &reply);
  // Act upon reply
  if (!status.ok()) {
    std::cout << "FAILED: " << status.error_message() << std::endl;
    return;
  }
  std::ofstream ofs_time(path.substr(0, path.find_first_of('.')) + "_remote.time", std::ios::binary);
  if (!reply.profiles().SerializeToOstream(&ofs_time)) {
    MS_LOG(EXCEPTION) << "Client: failed to serialize local GraphProfile";
  }
  ofs_time.close();
}

py::object OffloadingClient::PredictInner(const size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes, const std::unordered_map<std::string, py::object> &kwargs) {
  bool is_full_offloading = (lg_idx == 0);
  bool is_full_local = cut_nodes.empty();
  auto cut_nodes_label = GetCutLabel(cut_nodes);

  log_file << "CUT\t" << cut_nodes_label;

  if (!context_cache_.FindContext(cut_nodes_label)) {
    std::vector<std::string> output_names;
    if (!is_full_offloading) {
      auto cnode_list = latency_graph_manager_.GenerateKernelGraphSegmentClient(lg_idx, cut_nodes);
      auto graph_id = cost_graph_->GenerateKernelGraphFromSegment(session_impl_, cost_graph_->GetFullKernelGraph(), cnode_list, output_names);
      auto graph = session_impl_->GetGraph(graph_id);
      context_cache_.AddContext(lg_idx, cut_nodes_label, graph_id, graph, output_names);
    } else {
      context_cache_.AddContext(lg_idx, cut_nodes_label, kInvalidGraphId, nullptr, output_names);
    }
  }

  auto &ctx = context_cache_.GetContext(cut_nodes_label);
  if (!ProcessArg(kwargs, ctx, is_full_offloading)) {
    MS_LOG(EXCEPTION) << "Client: process args for execution failed";
    return py::none();
  }
  // Do inference
  tensor::TensorPtrList ret;
  if (!is_full_offloading) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(ctx.graph_id, ctx.input_tensors, &outputs);
    // cast return values
    ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed, no output";
      return py::none();
    }
    if (is_full_local) {
      for (auto &r : ret) {
        r->data_sync();
      }
      return BaseRefToPyData(outputs);
    }
  } else {
    ret = ctx.input_tensors;
  }

  PredictRequest request;
  request.set_lg_idx(lg_idx);
  request.set_cut_point(cut_nodes_label);
  if (cfg_.is_quant) {
    for (size_t i = 0; i < ret.size(); ++i) {
      std::vector<ShapeVector> tensor_shape = {ret[i]->shape()};
      tensor::TensorPtrList q_inputs;
      q_inputs.emplace_back(ret[i]);

      auto q_graph = quantizer_->GetKernelGraphByShape(tensor_shape);
      auto q_ret = quantizer_->Quantize(q_graph, q_inputs);
      if (q_ret.empty()) {
        MS_LOG(EXCEPTION) << "Client: quantization failed";
        return py::none();
      }
      // add quantized res & min & max to proto
      auto &q_tensor = q_ret[0];
      auto &max_val = q_ret[1];
      auto &min_val = q_ret[2];
      
      offloading_serving::QuantTensorProto *q_tensor_proto = request.add_tensor();
      auto tensor_proto = q_tensor_proto->mutable_tensor();
      (void)q_tensor->data_sync(); // must sync if data transmission is needed
      
      
      // auto q_start = TIMESTAMP();


      // auto q_start = TIMESTAMP();

      auto enable_cps = latency_graph_manager_.CheckEnableCps(ctx.output_names[i]);
      if (cfg_.is_compress && enable_cps) {
        TensorToProtoTensorCompressed(compressor_, q_tensor, tensor_proto);
      } else {
        TensorToProtoTensor(q_tensor, tensor_proto);
      }
      tensor_proto->set_name(ctx.output_names[i]);

      auto max_val_proto = q_tensor_proto->mutable_max();
      (void)max_val->data_sync();
      TensorToProtoTensor(max_val, max_val_proto);

      auto min_val_proto = q_tensor_proto->mutable_min();
      (void)min_val->data_sync();
      TensorToProtoTensor(min_val, min_val_proto);
    }
  } else {
    for (size_t i = 0; i < ret.size(); ++i) {
      offloading_serving::QuantTensorProto *q_tensor_proto = request.add_tensor();
      auto tensor_proto = q_tensor_proto->mutable_tensor();
      (void)ret[i]->data_sync(); // must sync if data transmission is needed
      if (cfg_.is_compress) {
        TensorToProtoTensorCompressed(compressor_, ret[i], tensor_proto);
      } else {
        TensorToProtoTensor(ret[i], tensor_proto);
      }
      tensor_proto->set_name(ctx.output_names[i]);
    }
  }
  // Container for the reply
  PredictReply reply;
  // Client context
  grpc::ClientContext context;
  // Do RPC
  grpc::Status status = stub_->Predict(&context, request, &reply);
  
  if (status.ok()) {
    if (reply.tensor_size() > 1) {
      py::tuple ret_tuple(reply.tensor_size());
      for (int i = 0; i < reply.tensor_size(); ++i) {
        auto ret_tensor = ProtoTensorToTensor(reply.tensor(i));
        ret_tuple[i] = TensorToPyData(ret_tensor);
      }
      return ret_tuple;
    } else if (reply.tensor_size() == 1) {
      auto ret_tensor = ProtoTensorToTensor(reply.tensor(0));
      auto ret_val = TensorToPyData(ret_tensor);
      return ret_val;
    }
  }
  return py::none();
}

bool OffloadingClient::ProcessArg(const std::unordered_map<std::string, py::object> &kwargs, OffloadingContext& ctx, const bool is_full_offloading) {
  std::unordered_map<std::string, tensor::TensorPtr> input_tensor_map;
  for (auto &item : kwargs) {
    py::object arg = item.second;
    auto tensor = PyDataToTensor(arg);
    if (!tensor) {
      MS_LOG(ERROR) << "Client: input arg convertion failed";
      return false;
    }
    input_tensor_map[item.first] = tensor;
  }

  auto& inputs = ctx.input_tensors;
  auto& output_names = ctx.output_names;
  if (is_full_offloading) {
    inputs.clear();
    output_names.clear();
    for (auto &it : input_tensor_map) {
      inputs.push_back(it.second);
      output_names.push_back(it.first);
    }
    return true;
  }

  auto& input_idx_map = ctx.input_name_list_idx_map;
  MS_EXCEPTION_IF_NULL(ctx.graph);
  auto& input_nodes = ctx.graph->inputs();
  if (inputs.empty()) {
    // first inference of this ctx
    for (size_t i = 0; i < input_nodes.size(); ++i) {
      MS_EXCEPTION_IF_NULL(input_nodes[i]);
      auto input_param_ptr = (input_nodes[i])->cast<ParameterPtr>();
      if (input_param_ptr->has_default()) {
        if (!input_param_ptr->default_param()->isa<tensor::Tensor>()) {
          MS_LOG(EXCEPTION) << "Client: Parameter[" << input_param_ptr->ToString()
                            << "] is not initialized, need to call `.init_data()`";
          return false;
        }
        auto input_tensor_ptr = input_param_ptr->default_param()->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(input_tensor_ptr);
        inputs.push_back(input_tensor_ptr);
      } else {
        auto name = input_param_ptr->DebugString(0);
        if (input_tensor_map.count(name) != 0) {
          auto input_tensor_ptr = input_tensor_map[name];
          // shape check
          if (!CompareInput(input_tensor_ptr, input_param_ptr)) {
            MS_LOG(EXCEPTION) << "Client: shape or type of args[" << name << "] is not compatible with param [" << name << "]";
            return false;
          }
          inputs.push_back(input_tensor_ptr);
          input_idx_map[name] = i;
        } else {
          MS_LOG(EXCEPTION) << "Client: Cannot find input arg for Parameter named: " << name;
          return false;
        }
      }
    }
  } else {
    for (auto& it : input_idx_map) {
      if (input_tensor_map.count(it.first) != 0) {
        auto input_tensor_ptr = input_tensor_map[it.first];
        auto input_param_ptr = (input_nodes[it.second])->cast<ParameterPtr>();
        // shape check
        if (!CompareInput(input_tensor_ptr, input_param_ptr)) {
          MS_LOG(EXCEPTION) << "Client: shape or type of args[" << it.first << "] is not compatible with param [" << it.first << "]";
          return false;
        }
        inputs[it.second] = input_tensor_ptr;
      } else {
        MS_LOG(EXCEPTION) << "Client: Cannot find input arg for Parameter named: " << it.first;
        return false;
      }
    }
  }
  return true;
}

void OffloadingClient::MeasureBandWidth() {
  char tmp[1048576];
  memset(tmp, 0, sizeof(tmp));
  while (is_measure_running_.load()) {
    if (!measure_mtx_.try_lock()) {
      std::this_thread::sleep_for(100ms);
      continue;
    } else {
      measure_mtx_.unlock();
    }
    size_t send_size = 1 << 20;

    mtx_.lock();
    if (!up_buf_.IsEmpty()) {
      auto avg_hist = up_buf_.GetAvgValue();
      int avg_hist_round = static_cast<size_t>(avg_hist * 8);
      avg_hist_round = avg_hist_round != 0 ? avg_hist_round : 1;
      send_size = static_cast<size_t>((double)avg_hist_round * (1 << 17));
      if (send_size > (1 << 20)) {
        send_size = 1 << 20;
      }
    }
    mtx_.unlock();
    double send_mb_size;
    grpc::ClientContext context;
    SimpleRequest request;
    SimpleReply reply;
    request.add_dumb(tmp, sizeof(char) * send_size);

    auto start_time = TIMESTAMP();
    grpc::Status status = stub_->TestUploadBandwidth(&context, request, &reply);
    if (!status.ok()) {
      MS_LOG(EXCEPTION) << "Client: connection failed when measuring bandwidth";
      return;
    }
    auto end_time = TIMESTAMP();
    auto cost = end_time - start_time;
    
    send_mb_size = ((double)request.ByteSize()) / (1 << 20);
    double bdw = send_mb_size / (cost * 1.0e-6);
    mtx_.lock();
    up_buf_.Push(bdw);
    mtx_.unlock();
    int64_t sleep_time = cost >= cfg_.measure_interval * 1000 ? 0 : cfg_.measure_interval * 1000 - cost;
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
  }
}

void OffloadingClient::MeasureLoad() {
  while (is_measure_running_.load()) {
    if (!measure_mtx_.try_lock()) {
      std::this_thread::sleep_for(100ms);
      continue;
    } else {
      measure_mtx_.unlock();
    }

    grpc::ClientContext context;
    SimpleRequest request;
    SimpleReply reply;
    request.add_dumb("");

    auto start_time = TIMESTAMP();
    grpc::Status status = stub_->TestUploadBandwidth(&context, request, &reply);
    if (!status.ok()) {
      MS_LOG(EXCEPTION) << "Client: connection failed when measuring bandwidth";
      return;
    }
    auto end_time = TIMESTAMP();
    auto cost = end_time - start_time;

    mtx_.lock();
    factor_buf_.Push(reply.factor());
    q_time_buf_.Push(reply.q_time());
    mtx_.unlock();
    int64_t sleep_time = cost >= cfg_.measure_interval * 1000 ? 0 : cfg_.measure_interval * 1000 - cost;
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
  }
}

Status OffloadingClient::InitEnv() {
  if (is_init_) {
    MS_LOG(WARNING) << "Env already initialized.";
    return SUCCESS;
  }
  // Create backend session
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = 0;
  if (device_target == kGPUDevice) {
    device_target = kGpuInferenceDevice;
  }
  session_impl_ = session::SessionFactory::Get().Create(device_target);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << device_target
                  << " is available.";
    return FAILED;
  }
  session_impl_->Init(device_id);
  is_init_ = true;
  return SUCCESS;
}

Status OffloadingClient::FinalizeEnv() {
  if (!is_init_) {
    MS_LOG(WARNING) << "Never initialize before.";
    return SUCCESS;
  }

  MS_LOG(INFO) << "Start finalize env";
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();

  is_init_ = false;
  MS_LOG(INFO) << "End finalize env";
  return SUCCESS;
}

Status OffloadingClient::CompileGraph(const FuncGraphPtr &func_graph, GraphId &graph_id) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    graph_id = session_impl_->CompileGraph(NOT_NULL(func_graph));
    return SUCCESS;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return FAILED;
  }
}

int OffloadingClient::PrintCNodeOrder() {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Client: graph not loaded, please call deploy()";
    return -1;
  }
  CNodePtrList full_ordered_cnodes = cost_graph_->GetFullKernelGraph()->execution_order();
  for (size_t i = 0; i < full_ordered_cnodes.size(); ++i) {
    auto node = full_ordered_cnodes[i]->cast<CNodePtr>();
    auto name = AnfAlgo::GetCNodeName(node);
    std::cout << "CNode[" << i << "]: " << full_ordered_cnodes[i]->fullname_with_scope() << std::endl;
  }
  return full_ordered_cnodes.size();
}

bool OffloadingClient::ProfileGraph(std::unordered_map<std::string, float> &result) {
  // prepare profiler
  std::shared_ptr<profiler::Profiler> profiler;
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#if ENABLE_GPU  
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else if (device_target == kGPUDevice) {
    profiler = profiler::gpu::GPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Client: offloading does not support devices except CPU and GPU for profiling";
    return false;
  }
#else 
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Client: offloading does not support devices except CPU for profiling";
    return false;
  }
#endif
  // prepare fake input args
  tensor::TensorPtrList input_tensors;
  auto& full_graph_ctx = context_cache_.GetContext("");
  auto& graph_id = full_graph_ctx.graph_id;
  auto& input_nodes = full_graph_ctx.graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto input_param_ptr = (input_nodes[i])->cast<ParameterPtr>();
    if (input_param_ptr->has_default()) {
      if (!input_param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Client: Parameter[" << input_param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
        return false;
      }
      auto input_tensor_ptr = input_param_ptr->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(input_tensor_ptr);
      input_tensors.push_back(input_tensor_ptr);
    } else {
      ShapeVector shape;
      auto param_shape = AnfAlgo::GetOutputDeviceShape(input_param_ptr, 0);
      (void)std::transform(param_shape.begin(), param_shape.end(), std::back_inserter(shape), 
                           [](const size_t dim) { return static_cast<int64_t>(dim); });
      auto dtype = AnfAlgo::GetSelectKernelBuildInfo(input_param_ptr)->GetOutputDeviceType(0);
      auto input_tensor_ptr = std::make_shared<tensor::Tensor>(dtype, shape);
      auto *data_buf = reinterpret_cast<uint8_t *>(input_tensor_ptr->data_c());
      MS_EXCEPTION_IF_NULL(data_buf);
      auto ret = memset_s(data_buf, input_tensor_ptr->data().nbytes(), 0, input_tensor_ptr->data().nbytes());
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "Client: memset_s error for building Tensor for profile, errorno " << ret;
        return false;
      }
      input_tensors.push_back(input_tensor_ptr);
    }
  }

  // warm-up
  for (int i = 0; i < cfg_.profile_times / 2; ++i) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed when profiling, no output";
      return false;
    }
    for (auto &t : ret) {
      t->data_sync();
    }
  }
  profiler->Init("");
  profiler->StepProfilingEnable(true);
  // inference
  for (int i = 0; i < cfg_.profile_times; ++i) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed when profiling, no output";
      return false;
    }
    for (auto &t : ret) {
      t->data_sync();
    }
  }
  // get result and stop profiler
  profiler->StopWithoutWriteFile(result);
  return true;
}

bool OffloadingClient::ProfileGraph(KernelGraphPtr &kg, std::unordered_map<std::string, float> &result) {
  // prepare profiler
  std::shared_ptr<profiler::Profiler> profiler;
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#if ENABLE_GPU  
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else if (device_target == kGPUDevice) {
    profiler = profiler::gpu::GPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Client: offloading does not support devices except CPU and GPU for profiling";
    return false;
  }
#else 
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Client: offloading does not support devices except CPU for profiling";
    return false;
  }
#endif
  // prepare fake input args
  tensor::TensorPtrList input_tensors;
  auto graph_id = kg->graph_id();
  auto& input_nodes = kg->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto input_param_ptr = (input_nodes[i])->cast<ParameterPtr>();
    if (input_param_ptr->has_default()) {
      if (!input_param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Client: Parameter[" << input_param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
        return false;
      }
      auto input_tensor_ptr = input_param_ptr->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(input_tensor_ptr);
      input_tensors.push_back(input_tensor_ptr);
    } else {
      ShapeVector shape;
      auto param_shape = AnfAlgo::GetOutputDeviceShape(input_param_ptr, 0);
      (void)std::transform(param_shape.begin(), param_shape.end(), std::back_inserter(shape), 
                           [](const size_t dim) { return static_cast<int64_t>(dim); });
      auto dtype = AnfAlgo::GetSelectKernelBuildInfo(input_param_ptr)->GetOutputDeviceType(0);
      auto input_tensor_ptr = std::make_shared<tensor::Tensor>(dtype, shape);
      auto *data_buf = reinterpret_cast<uint8_t *>(input_tensor_ptr->data_c());
      MS_EXCEPTION_IF_NULL(data_buf);
      auto ret = memset_s(data_buf, input_tensor_ptr->data().nbytes(), 0, input_tensor_ptr->data().nbytes());
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "Client: memset_s error for building Tensor for profile, errorno " << ret;
        return false;
      }
      input_tensors.push_back(input_tensor_ptr);
    }
  }

  // warm-up
  for (int i = 0; i < cfg_.profile_times / 2; ++i) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed when profiling, no output";
      return false;
    }
    for (auto &t : ret) {
      t->data_sync();
    }
  }
  profiler->Init("");
  profiler->StepProfilingEnable(true);
  // inference
  for (int i = 0; i < cfg_.profile_times; ++i) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed when profiling, no output";
      return false;
    }
    for (auto &t : ret) {
      t->data_sync();
    }
  }
  // get result and stop profiler
  profiler->StopWithoutWriteFile(result);
  return true;
}

void OffloadingClient::ConvertProfileResToCSV(const std::string &path) {
  GraphProfile profile_proto;
  std::ifstream ifs(path, std::ios::binary);
  if (!profile_proto.ParseFromIstream(&ifs)) {
    MS_LOG(EXCEPTION) << "Client: failed to load cloud time map proto file at " << path;
  }
  ifs.close();

  for (const auto& batch_entry : profile_proto.entries()) {
    std::ofstream ofs(path + "_" + std::to_string(batch_entry.batch_size()) + ".csv");
    if (!ofs.is_open()) {
      MS_LOG(WARNING) << "Open file '" << path + "_" + std::to_string(batch_entry.batch_size()) + ".csv" << "' failed!";
      return;
    }
    ofs << "op,time\n";
    for (const auto& profile_entry : batch_entry.profile()) {
      ofs << "\"" << profile_entry.name() << "\"," << profile_entry.time() << "\n";
    }
    ofs.close();
  }
}

void OffloadingClient::AsyncCompleteFakePredict() {
  void *got_tag;
  bool ok = false;
  // Block until the next result is available in the completion queue "cq".
  while (cq_.Next(&got_tag, &ok)) {
    AsyncPredictCall* call = static_cast<AsyncPredictCall*>(got_tag);
    MS_EXCEPTION_IF_CHECK_FAIL(ok, "complete async predict call failed");
    // auto &reply = call->reply;
    // log_file << reply.factor() << "\t" << reply.q_time() << std::endl;
    
    if (!call->status.ok()) {
      MS_LOG(EXCEPTION) << "Async FakePredict RPC failed";
    }
    delete call;
  }
}

void OffloadingClient::PartitionDecisionTest(double bdw, double q_time, double load_factor) {
  if (!is_deployed_) {
      MS_LOG(WARNING) << "Client: model is not loaded, please call Deploy";
      return;
    }

    LatencyGraph::PartitionResult res;
    auto start_time = TIMESTAMP();
    auto lg_idx = latency_graph_manager_.PartitionDecision(bdw, q_time, load_factor, res);
    auto end_time = TIMESTAMP();
    auto pda_time = end_time - start_time;
    std::cout << "PDA time: " << pda_time << " us" << std::endl;
    auto cut_nodes_label = GetCutLabel(res.best_cut_nodes_);
    std::cout << lg_idx << " : " << cut_nodes_label << std::endl;
    log_file << bdw << '\t' << q_time << '\t' << load_factor << '\t' << lg_idx << '\t' << cut_nodes_label << std::endl;
}

void OffloadingClient::FakeDeploy(const std::string &path) {
  if (is_deployed_fake_ || is_deployed_) return;
  DeployRequest request;
  request.set_file_name(path);
  // Container for the reply
  DeployReply reply;
  // Client context
  grpc::ClientContext context;
  // Do RPC
  grpc::Status status = stub_->Deploy(&context, request, &reply);
  // Act upon reply
  if (!status.ok()) {
    std::cout << "FAILED: " << status.error_message() << std::endl;
    return;
  }
  fake_pred_completion_thread_ = std::thread(&OffloadingClient::AsyncCompleteFakePredict, this);
  
  log_file.open("client.log");
  if (!log_file.is_open()) {
    MS_LOG(EXCEPTION) << "Client: open file client.log failed!";
  }
  is_deployed_fake_ = true;
}

void OffloadingClient::FakeStaticPredictTest(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  if (!is_deployed_fake_ && !is_deployed_) {
    MS_LOG(WARNING) << "Client: model is not loaded, please call FakeDeploy";
    return;
  }

  auto cut_nodes_label = GetCutLabel(cut_nodes_names);
  // Prepare request data
  PredictRequest request;
  request.set_lg_idx(lg_idx);
  request.set_cut_point(cut_nodes_label);
  for (auto &item : kwargs) {
    py::object arg = item.second;
    if (py::isinstance<py::list>(arg)) {
      auto input_list = py::cast<py::list>(arg);
      if (input_list.size() == 1) {
        auto tensor = PyDataToTensor(input_list[0]);
        if (!tensor) {
          MS_LOG(ERROR) << "Client: input arg convertion failed";
        }
        offloading_serving::QuantTensorProto *q_tensor_proto = request.add_tensor();
        auto tensor_proto = q_tensor_proto->mutable_tensor();
        TensorToProtoTensor(tensor, tensor_proto);
        tensor_proto->set_name(item.first);
      } else if (input_list.size() == 3) {
        auto tensor = PyDataToTensor(input_list[0]);
        if (!tensor) {
          MS_LOG(ERROR) << "Client: input arg convertion failed";
        }
        auto max_val = PyDataToTensor(input_list[1]);
        if (!max_val) {
          MS_LOG(ERROR) << "Client: input arg convertion failed";
        }
        auto min_val = PyDataToTensor(input_list[2]);
        if (!min_val) {
          MS_LOG(ERROR) << "Client: input arg convertion failed";
        }
        offloading_serving::QuantTensorProto *q_tensor_proto = request.add_tensor();
        auto tensor_proto = q_tensor_proto->mutable_tensor();
        TensorToProtoTensor(tensor, tensor_proto);
        auto max_val_proto = q_tensor_proto->mutable_max();
        TensorToProtoTensor(max_val, max_val_proto);
        auto min_val_proto = q_tensor_proto->mutable_min();
        TensorToProtoTensor(min_val, min_val_proto);
        tensor_proto->set_name(item.first);
      } else {
        MS_LOG(ERROR) << "Client: if need quantization, the inputs for " << item.first << " should have exactly 1 or 3 elements";
      }
    } else {
      auto tensor = PyDataToTensor(arg);
      if (!tensor) {
        MS_LOG(ERROR) << "Client: input arg convertion failed";
      }
      offloading_serving::QuantTensorProto *q_tensor_proto = request.add_tensor();
      auto tensor_proto = q_tensor_proto->mutable_tensor();
      TensorToProtoTensor(tensor, tensor_proto);
      tensor_proto->set_name(item.first);
    }
  }
  AsyncPredictCall *call = new AsyncPredictCall;
  call->response_reader = stub_->PrepareAsyncPredict(&call->context, request, &cq_);
  call->response_reader->StartCall();
  call->response_reader->Finish(&call->reply, &call->status, (void*)call);
}

py::object OffloadingClient::PartialExecute(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Client: model is not loaded, please call Deploy";
    return py::none();
  }

  // find cut nodes by name
  std::unordered_set<CostGraph::NodePtr> cut_nodes;
  cost_graph_->GetNodesByName(cut_nodes_names, cut_nodes);
  bool is_full_offloading = (lg_idx == 0);
  bool is_full_local = cut_nodes.empty();
  auto cut_nodes_label = GetCutLabel(cut_nodes);
  if (!context_cache_.FindContext(cut_nodes_label)) {
    std::vector<std::string> output_names;
    if (!is_full_offloading) {
      auto cnode_list = latency_graph_manager_.GenerateKernelGraphSegmentClient(lg_idx, cut_nodes);
      auto graph_id = cost_graph_->GenerateKernelGraphFromSegment(session_impl_, cost_graph_->GetFullKernelGraph(), cnode_list, output_names);
      auto graph = session_impl_->GetGraph(graph_id);
      context_cache_.AddContext(lg_idx, cut_nodes_label, graph_id, graph, output_names);
    } else {
      context_cache_.AddContext(lg_idx, cut_nodes_label, kInvalidGraphId, nullptr, output_names);
    }
  }

  auto &ctx = context_cache_.GetContext(cut_nodes_label);
  if (!ProcessArg(kwargs, ctx, is_full_offloading)) {
    MS_LOG(EXCEPTION) << "Client: process args for execution failed";
    return py::none();
  }

  // Do inference
  tensor::TensorPtrList ret;
  if (!is_full_offloading) {
    VectorRef outputs;
    session_impl_->RunGraphAsync(ctx.graph_id, ctx.input_tensors, &outputs);
    // cast return values
    ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Client: convert to Tensor failed, no output";
      return py::none();
    }
    if (is_full_local) {
      return BaseRefToPyData(outputs);
    }
  } else {
    ret = ctx.input_tensors;
  }

  for (auto &r : ret) {
    r->data_sync();
  }
  if (cfg_.is_quant) {
    py::tuple ret_tuple(ret.size() * 3);
    for (size_t i = 0; i < ret.size(); ++i) {
      auto before_quant = TIMESTAMP();
      
      std::vector<ShapeVector> tensor_shape = {ret[i]->shape()};
      tensor::TensorPtrList q_inputs;
      q_inputs.emplace_back(ret[i]);

      auto q_graph = quantizer_->GetKernelGraphByShape(tensor_shape);
      auto q_ret = quantizer_->Quantize(q_graph, q_inputs);
      if (q_ret.empty()) {
        MS_LOG(EXCEPTION) << "Client: quantization failed";
        return py::none();
      }
      (void)q_ret[0]->data_sync();

      ret_tuple[3 * i] = TensorToPyData(q_ret[0]);
      (void)q_ret[1]->data_sync();
      ret_tuple[3 * i + 1] = TensorToPyData(q_ret[1]);
      (void)q_ret[2]->data_sync();
      ret_tuple[3 * i + 2] = TensorToPyData(q_ret[2]);
      
      auto after_quant = TIMESTAMP();
      
      log_file << after_quant - before_quant << std::endl;
      if (cfg_.is_compress) {
        offloading_serving::TensorProto tensor_proto;
        TensorToProtoTensorCompressed(compressor_, q_ret[0], &tensor_proto);
      } else {
        offloading_serving::TensorProto tensor_proto;
        TensorToProtoTensor(q_ret[0], &tensor_proto);
      }
    }
    return ret_tuple;
  } else {
    py::tuple ret_tuple(ret.size());
    for (size_t i = 0; i < ret.size(); ++i) {
      (void)ret[i]->data_sync();
      ret_tuple[i] = TensorToPyData(ret[i]);
    }
    return ret_tuple;
  }
  return py::none();
}

py::object OffloadingClient::ConvertProtoTensorFromFile(const std::string &path) {
  offloading_serving::TensorProto tensor_proto;
  std::ifstream ifs(path, std::ios::binary);
  if (!tensor_proto.ParseFromIstream(&ifs)) {
    MS_LOG(EXCEPTION) << "Client: failed to load cloud time map proto file at " << path.substr(0, path.find_first_of('.')) + "_remote.time";
  }
  ifs.close();
  auto tensor = ProtoTensorToTensor(tensor_proto);
  return TensorToPyData(tensor);
}

py::object OffloadingClient::UnitTest(const std::unordered_map<std::string, py::object> &kwargs) {
  auto path = kwargs.begin()->first;
  GraphProfile local_prof_map_proto, cloud_time_map_proto;
  std::ifstream ifs_l(path.substr(0, path.find_first_of('.')) + "_local.prof", std::ios::binary);
  if (!local_prof_map_proto.ParseFromIstream(&ifs_l)) {
    MS_LOG(EXCEPTION) << "Client: failed to load cloud time map proto file at " << path.substr(0, path.find_first_of('.')) + "_local.prof";
  }
  ifs_l.close();

  std::ifstream ifs(path.substr(0, path.find_first_of('.')) + "_remote.time", std::ios::binary);
  if (!cloud_time_map_proto.ParseFromIstream(&ifs)) {
    MS_LOG(EXCEPTION) << "Client: failed to load cloud time map proto file at " << path.substr(0, path.find_first_of('.')) + "_remote.time";
  }
  ifs.close();

  std::unordered_map<std::string, float> local_prof_map, cloud_time_map;
  for (const auto& batch_entry : local_prof_map_proto.entries()) {
    if (batch_entry.batch_size() == 1) {
      for (const auto& profile_entry : batch_entry.profile()) {
        local_prof_map[profile_entry.name()] = profile_entry.time();
      }
      break;
    }
  }
  DumpExecTimeTSV(path.substr(0, path.find_first_of('.')) + "_prof.tsv", local_prof_map, 1);

  for (const auto& batch_entry : cloud_time_map_proto.entries()) {
    if (batch_entry.batch_size() == 1) {
      for (const auto& profile_entry : batch_entry.profile()) {
        cloud_time_map[profile_entry.name()] = profile_entry.time();
      }
      break;
    }
  }
  DumpExecTimeTSV(path.substr(0, path.find_first_of('.')) + "_time.tsv", cloud_time_map, 1);

  return py::none();
}

}
}
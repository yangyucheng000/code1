#include <string>

#include "batch_offloading_server.h"

namespace mindspore {
namespace offloading {

Status BatchOffloadingServer::StartGrpcServer(const std::string &ip, uint32_t grpc_port, OffloadingServerConfig config,
                                              int max_msg_mb_size = gRpcDefaultMsgMBSize) {
  if (grpc_async_server_) {
    MS_LOG(ERROR) << "Serving Error: BatchOffloading gRPC server is already running";
    return FAILED;
  }
  if (max_msg_mb_size > gRpcMaxMBMsgSize) {
    MS_LOG(WARNING) << "The maximum Serving gRPC message size is 512MB and will be updated from " << max_msg_mb_size
                    << "MB to 512MB";
    max_msg_mb_size = gRpcMaxMBMsgSize;
  }
  grpc_async_server_ = std::make_shared<BatchOffloadingGrpcServer>(config);
  std::string socket_address = ip + ":" + std::to_string(grpc_port);
  return grpc_async_server_->Start(socket_address, max_msg_mb_size, "Serving gRPC");
}

void BatchOffloadingServer::Stop() {
  grpc_async_server_->Stop();
  grpc_async_server_ = nullptr;
}

BatchOffloadingServer &BatchOffloadingServer::GetInstance() {
  static BatchOffloadingServer server;
  return server;
}

void BatchOffloadingServiceImpl::PredictAsync(const offloading_serving::PredictRequest *request,
                                              offloading_serving::PredictReply *reply, PredictOnFinish on_finish) {
  if (!is_deployed_) {
    MS_LOG(WARNING) << "Server: model is not loaded, please call Deploy";
    on_finish(reply, grpc::Status::CANCELLED);
    return;
  }
  // To scheduler
  sched_->ProcessRequest(const_cast<offloading_serving::PredictRequest *>(request), reply, on_finish);
}

grpc::Status BatchOffloadingServiceImpl::Deploy(const offloading_serving::DeployRequest *request,
                                                offloading_serving::DeployReply *reply) {
  if (is_init_ && is_deployed_) return grpc::Status::OK;
  // Init scheduler & executor
  if (!is_init_) {
    sched_->Init(cfg_, request->file_name());
    is_init_ = true;
  }
  reply->set_s("Success");
  is_deployed_ = true;
  return grpc::Status::OK;
}

grpc::Status BatchOffloadingServiceImpl::Profile(const offloading_serving::ProfileRequest *request,
                                                 offloading_serving::ProfileReply *reply) {
  // Init env
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = 0;
  if (device_target == kGPUDevice) {
    device_target = kGpuInferenceDevice;
  }
  session::SessionPtr session_impl = session::SessionFactory::Get().Create(device_target);
  if (session_impl == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << device_target << " is available.";
    return grpc::Status::CANCELLED;
  }
  session_impl->Init(device_id);
  // Load full model as FuncGraph
  std::string path = request->file_name();
  MindIRLoader model_loader;
  FuncGraphPtr full_func_graph = model_loader.LoadMindIR(path);
  if (full_func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Server: load MindIR model failed";
    return grpc::Status::CANCELLED;
  }

  std::unordered_map<size_t, std::unordered_map<std::string, float>> graph_profile;
  std::unordered_map<size_t, std::unordered_map<std::string, float>> cloud_time_map;
  for (size_t bsz = 1; bsz <= cfg_.max_batch_size; ++bsz) {
    // change batch size
    std::cout << "Profile graph with batch size = " << bsz << std::endl;
    if (bsz != 1) {
      const auto &inputs = full_func_graph->get_inputs();
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &param = inputs[i];
        auto shape_ptr = std::dynamic_pointer_cast<abstract::Shape>(param->Shape());
        if (shape_ptr == nullptr) {
          MS_LOG(ERROR) << "inputs " << i << " is not supported to resize, debug string: " << param->DebugString();
          return grpc::Status::CANCELLED;
        }
        auto tmp_shape = shape_ptr->shape();
        tmp_shape[0] = bsz;
        shape_ptr->set_shape(tmp_shape);
      }
    }
    // compile & get graph
    GraphId full_graph_id;
    KernelGraphPtr full_graph;
    try {
      if (bsz != 1) {
        full_graph_id = session_impl->CompileGraphWithInferShape(NOT_NULL(full_func_graph));
      } else {
        full_graph_id = session_impl->CompileGraph(NOT_NULL(full_func_graph));
      }
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Server: compile full graph failed: " << e.what();
      return grpc::Status::CANCELLED;
    }

    full_graph = session_impl->GetGraph(full_graph_id);
    if (full_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Server: GetGraph from session failed";
      return grpc::Status::CANCELLED;
    }
    FuncGraphManagerPtr manager = MakeManager({full_graph});
    if (manager) {
      manager->AddFuncGraph(full_graph);
      full_graph->set_manager(manager);
    }
    // Profile KernelGraph and construct CostGraph
    graph_profile[bsz] = std::unordered_map<std::string, float>();
    if (!ProfileGraph(full_graph_id, full_graph, session_impl, graph_profile[bsz])) {
      MS_LOG(EXCEPTION) << "Server: profile KernelGraph failed";
      return grpc::Status::CANCELLED;
    }
    cloud_time_map[bsz] = GenerateTimeMapWithRenaming(full_graph, graph_profile[bsz]);
  }

  // construct proto and write to file
  GraphProfile profile_proto;
  for (size_t bsz = 1; bsz <= cfg_.max_batch_size; ++bsz) {
    GraphProfile::BatchProfileEntry *batch_profile_entry = profile_proto.add_entries();
    batch_profile_entry->set_batch_size(bsz);
    for (const auto &entry : graph_profile[bsz]) {
      GraphProfile::ProfileEntry *profile_entry = batch_profile_entry->add_profile();
      profile_entry->set_name(entry.first);
      profile_entry->set_time(entry.second);
    }
  }
  std::ofstream ofs(path.substr(0, path.find_first_of('.')) + "_remote.prof", std::ios::binary);
  if (!profile_proto.SerializeToOstream(&ofs)) {
    MS_LOG(EXCEPTION) << "Server: failed to serialize remote GraphProfile";
  }
  ofs.close();

  GraphProfile *profiles = reply->mutable_profiles();
  for (size_t bsz = 1; bsz <= cfg_.max_batch_size; ++bsz) {
    GraphProfile::BatchProfileEntry *batch_profile_entry = profiles->add_entries();
    batch_profile_entry->set_batch_size(bsz);
    for (const auto &entry : cloud_time_map[bsz]) {
      GraphProfile::ProfileEntry *profile_entry = batch_profile_entry->add_profile();
      profile_entry->set_name(entry.first);
      profile_entry->set_time(entry.second);
    }
  }

  std::ofstream ofs_time(path.substr(0, path.find_first_of('.')) + "_remote.time", std::ios::binary);
  if (!profiles->SerializeToOstream(&ofs_time)) {
    MS_LOG(EXCEPTION) << "Server: failed to serialize remote GraphProfile";
  }
  ofs_time.close();

  return grpc::Status::OK;
}

bool BatchOffloadingServiceImpl::ProfileGraph(GraphId graph_id, KernelGraphPtr &graph, session::SessionPtr &session_impl, std::unordered_map<std::string, float> &result) {
  // prepare profiler
  std::shared_ptr<profiler::Profiler> profiler;
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#if ENABLE_GPU  
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else if (device_target == kGPUDevice) {
    profiler = profiler::gpu::GPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Server: offloading does not support devices except CPU and GPU for profiling";
    return false;
  }
#else 
  if (device_target == kCPUDevice) {
    profiler = profiler::cpu::CPUProfiler::GetInstance();
  } else {
    MS_LOG(EXCEPTION) << "Server: offloading does not support devices except CPU for profiling";
    return false;
  }
#endif
  // prepare fake input args
  tensor::TensorPtrList input_tensors;
  auto& input_nodes = graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto input_param_ptr = (input_nodes[i])->cast<ParameterPtr>();
    if (input_param_ptr->has_default()) {
      if (!input_param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Server: Parameter[" << input_param_ptr->ToString()
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
        MS_LOG(EXCEPTION) << "Server: memset_s error for building Tensor for profile, errorno " << ret;
        return false;
      }
      input_tensors.push_back(input_tensor_ptr);
    }
  }

  // warm-up
  for (int i = 0; i < cfg_.profile_times / 2; ++i) {
    VectorRef outputs;
    session_impl->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Server: convert to Tensor failed when profiling, no output";
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
    session_impl->RunGraphAsync(graph_id, input_tensors, &outputs);
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Server: convert to Tensor failed when profiling, no output";
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

grpc::Status BatchOffloadingServiceImpl::TestUploadBandwidth(const offloading_serving::SimpleRequest *request,
                                                             offloading_serving::SimpleReply *reply) {
  reply->add_dumb("");
  double factor, q_time;
  sched_->GetRuntimeProfile(factor, q_time);
  reply->set_factor(factor);
  reply->set_q_time(q_time);
  return grpc::Status::OK;
}

grpc::Status BatchOffloadingServiceImpl::TestDownloadBandwidth(const offloading_serving::SimpleRequest *request,
                                                               offloading_serving::SimpleReply *reply) {
  reply->add_dumb("");
  return grpc::Status::OK;
}

}  // namespace offloading
}  // namespace mindspore

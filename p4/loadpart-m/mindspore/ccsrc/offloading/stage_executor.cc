#include "stage_executor.h"

namespace mindspore {

namespace offloading {

tensor::TensorPtr MergeTensorPair(tensor::TensorPtr &tensor_a, tensor::TensorPtr &tensor_b) {
  MS_EXCEPTION_IF_NULL(tensor_a);
  MS_EXCEPTION_IF_NULL(tensor_b);
  
  tensor_a->Wait();
  tensor_b->Wait();

  auto shape_a = tensor_a->shape();
  auto shape_b = tensor_b->shape();
  bool is_same_shape_wo_bs = std::equal(shape_a.begin() + 1, shape_a.end(), shape_b.begin() + 1, shape_b.end());
  if (!is_same_shape_wo_bs) {
    MS_LOG(EXCEPTION) << "Server: shapes without batch dim are not the same";
    return nullptr;
  }
  if (tensor_a->data_type() != tensor_b->data_type()) {
    MS_LOG(EXCEPTION) << "Server: types of the two tensors are not the same";
    return nullptr;
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, 0);
  auto mem_manager = runtime_instance->GetMemoryManager();
  // allocate a new space from mem pool
  // create a new device address
  auto new_addr_ptr = mem_manager->MallocMemFromMemPool(tensor_a->Size() + tensor_b->Size(), false);
  auto new_device_addr = runtime_instance->CreateDeviceAddress(new_addr_ptr, tensor_a->Size() + tensor_b->Size(), tensor_a->device_info().format_, tensor_a->data_type());
  // create a new Tensor accordingly
  ShapeVector new_shape = shape_a;
  new_shape[0] += shape_b[0];
  auto tensor_a_b = std::make_shared<Tensor>(tensor_a->data_type(), new_shape);
  tensor_a_b->set_device_address(new_device_addr, false);
  // copy the two tensors to the new device async (by cudaMemcpyAsync or a kernel?)
  if (!new_device_addr->AsyncDeviceToDevice(tensor_a->device_address().get(), (size_t)0)) {
    MS_LOG(ERROR) << "Server: copy async failed";
    return nullptr;
  }

  if (!new_device_addr->AsyncDeviceToDevice(tensor_b->device_address().get(), tensor_a->Size())) {
    MS_LOG(ERROR) << "Server: copy async failed";
    return nullptr;
  }
  new_device_addr->set_from_mem_pool(true);
  new_device_addr->set_status(device::DeviceAddressStatus::kInDevice);
  tensor_a_b->set_sync_status(kNoNeedSync);
  // need free a/b?
  // tensor_a->device_address()->ClearDeviceMemory();
  // tensor_b->device_address()->ClearDeviceMemory();
  auto tensor_a_addr = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_a->device_address());
  MS_EXCEPTION_IF_NULL(tensor_a_addr);
  auto tensor_b_addr = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_b->device_address());
  MS_EXCEPTION_IF_NULL(tensor_b_addr);
  mem_manager->FreeMemFromMemPool(tensor_a_addr);
  tensor_a_addr->set_status(device::DeviceAddressStatus::kInDevice);
  mem_manager->FreeMemFromMemPool(tensor_b_addr);
  tensor_b_addr->set_status(device::DeviceAddressStatus::kInDevice);
  return tensor_a_b;
}

void LoadIntermidiateData(tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (!tensor->device_address()) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, 0);
    auto mem_manager = runtime_instance->GetMemoryManager();
    // allocate a new space from mem pool
    // create a new device address
    auto new_addr_ptr = mem_manager->MallocMemFromMemPool(tensor->Size(), false);
    auto new_device_addr = runtime_instance->CreateDeviceAddress(new_addr_ptr, tensor->Size(), tensor->device_info().format_, tensor->data_type());
    tensor->set_device_address(new_device_addr, false);
    if (!new_device_addr->SyncHostToDevice(tensor->Size(), tensor->data_c())) {
      MS_LOG(ERROR) << "Server: copy sync failed";
    }
  }
}

void StageExecutor::Init() {
  if (!is_running_.load()) {
    is_running_.store(true);
  }
  executor_ = std::thread(&StageExecutor::Run, this);
}

void StageExecutor::Stop() {
  if (is_running_.load()) {
    is_running_.store(false);
    executor_.join();
  }
}

void StageExecutor::Run() {
  while (is_running_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void PreprocessExecutor::EnqueuePayload(PayloadPtr& payload) {
  queue_.Enqueue(payload);
}

void PreprocessExecutor::Run() {
  while (is_running_.load()) {
    // payload->requests_ should be sorted according to the entry points
    auto payload = queue_.Dequeue();
    MS_EXCEPTION_IF_NULL(payload->plan_);

    auto dequeue_time = TIMESTAMP();

    // Decompress -> De-quantize all -> create a task
    TaskPtr task = std::unique_ptr<Task>(new Task());
    task->inputs.resize(payload->plan_->entry_list.size());
    ConstructTaskInputs(payload, task->inputs);
    // add plan
    task->stage_id = 1;
    task->plan = payload->plan_;
    std::copy(payload->session_ids_.begin(), payload->session_ids_.end(), std::back_inserter(task->session_ids));
    auto next_executor = std::dynamic_pointer_cast<InferenceExecutor>(next_executor_);
    if (!next_executor) {
      MS_LOG(ERROR) << "The next StageExecutor of PreprocessExecutor should be InferenceExecutor.";
      return;
    }

    task->req_q_times.insert(task->req_q_times.end(), payload->req_q_times_.begin(), payload->req_q_times_.end());
    // first stage queueing time
    task->q_times[0] = dequeue_time - payload->q_time_;
    // scheduler execution time
    task->e_times[0] = payload->e_time_;
    auto exec_finish_time = TIMESTAMP();
    // first stage execution time
    task->e_times[1] = exec_finish_time - dequeue_time;
    task->q_times[1] = exec_finish_time;

    next_executor->EnqueueTask(std::move(task));
  }
}

void PreprocessExecutor::ConstructTaskInputs(PayloadPtr& payload, std::vector<NameToQTensorMap>& inputs) {
  MS_EXCEPTION_IF_NULL(payload);
  auto &bsz_list = payload->plan_->bsz_list;
  // i for entry point index, j&k for traversing requests
  for (size_t i = 0, j = 0; i < bsz_list.size(); j += bsz_list[i++]) {
    auto &cur_bsz = bsz_list[i];
    // assume the ts-th tensors of requests between [j, j + cur_bsz) have the same dims/data_type/name 
    auto &r = payload->requests_[j];
    for (int ts = 0; ts < r->tensor_size(); ++ts) {
      // create a new tensor
      auto &first_q_tensor_proto = r->tensor(ts);
      auto &first_tensor_proto = first_q_tensor_proto.tensor();
      ShapeVector shape;
      for (int s = 0; s < first_tensor_proto.dims_size(); ++s) {
        shape.push_back(first_tensor_proto.dims(s));
      }
      shape[0] *= cur_bsz;

      auto dtype = GetTypeIdFromProtoTensor(first_tensor_proto);
      tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(dtype, shape);
      auto tensor_data_c = reinterpret_cast<uint8_t *>(tensor->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data_c);
      size_t single_batch_size = std::accumulate(first_tensor_proto.dims().begin(), first_tensor_proto.dims().end(), 1, std::multiplies<size_t>()) * kernel::UnitSizeInBytes(dtype);
      size_t remain_buf_size = tensor->data().nbytes();

      for (size_t k = 0; k < cur_bsz; ++k) {
        auto &q_tensor_proto = payload->requests_[j + k]->tensor(ts);
        auto &tensor_proto = q_tensor_proto.tensor();
        auto &init_data = tensor_proto.raw_data();
        if (tensor_proto.compressed()) {
          auto *init_data_c = reinterpret_cast<const void *>(init_data.data());
          MS_EXCEPTION_IF_NULL(init_data_c);
          auto *data_buf = reinterpret_cast<void *>(tensor_data_c);
          MS_EXCEPTION_IF_NULL(data_buf);
          auto ret = decompressor_.Decompress(init_data_c, init_data.size(), data_buf, remain_buf_size);
          if (ret < 0) {
            MS_LOG(EXCEPTION) << "Decompress error for building Tensor from TensorProto, errorno " << ret;
            return;
          }
          tensor_data_c += single_batch_size;
          remain_buf_size -= single_batch_size;
        } else {
          auto ret = memcpy_s(tensor_data_c, remain_buf_size, init_data.data(), init_data.size());
          if (ret != 0) {
            MS_LOG(EXCEPTION) << "memcpy_s error for building Tensor from TensorProto, errorno " << ret;
            return;
          }
          tensor_data_c += single_batch_size;
          remain_buf_size -= single_batch_size;
        }
      }
      // batch max & min value for quant tensors
      std::vector<tensor::TensorPtr> q_tensors;
      q_tensors.emplace_back(tensor);
      if (is_dequant_) {
        if (!first_q_tensor_proto.has_max()) {
          MS_LOG(EXCEPTION) << "Server: request proto has no quant max/min values in dequant mode";
        }
        auto &first_max_val_proto = first_q_tensor_proto.max(); // assume max is in the same shape&type with min
        ShapeVector m_shape;
        for (int s = 0; s < first_max_val_proto.dims_size(); ++s) {
          m_shape.push_back(first_max_val_proto.dims(s));
        }
        m_shape[0] *= cur_bsz;
        auto m_dtype = GetTypeIdFromProtoTensor(first_max_val_proto);
        tensor::TensorPtr max_tensor = std::make_shared<tensor::Tensor>(m_dtype, m_shape);
        tensor::TensorPtr min_tensor = std::make_shared<tensor::Tensor>(m_dtype, m_shape);
        auto max_tensor_data_c = reinterpret_cast<uint8_t *>(max_tensor->data_c());
        MS_EXCEPTION_IF_NULL(max_tensor_data_c);
        auto min_tensor_data_c = reinterpret_cast<uint8_t *>(min_tensor->data_c());
        MS_EXCEPTION_IF_NULL(min_tensor_data_c);
        size_t m_single_batch_size = first_max_val_proto.raw_data().size();
        size_t m_remain_buf_size = max_tensor->data().nbytes();

        for (size_t k = 0; k < cur_bsz; ++k) {
          auto &q_tensor_proto = payload->requests_[j + k]->tensor(ts);
          
          auto &max_val_proto = q_tensor_proto.max();
          auto &max_init_data = max_val_proto.raw_data();
          auto ret = memcpy_s(max_tensor_data_c, m_remain_buf_size, max_init_data.data(), max_init_data.size());
          if (ret != 0) {
            MS_LOG(EXCEPTION) << "memcpy_s error for building Tensor from TensorProto, errorno " << ret;
            return;
          }

          auto &min_val_proto = q_tensor_proto.min();
          auto &min_init_data = min_val_proto.raw_data();
          ret = memcpy_s(min_tensor_data_c, m_remain_buf_size, min_init_data.data(), min_init_data.size());
          if (ret != 0) {
            MS_LOG(EXCEPTION) << "memcpy_s error for building Tensor from TensorProto, errorno " << ret;
            return;
          }

          max_tensor_data_c += m_single_batch_size;
          min_tensor_data_c += m_single_batch_size;
          m_remain_buf_size -= m_single_batch_size;
        }

        q_tensors.emplace_back(max_tensor);
        q_tensors.emplace_back(min_tensor);
      }
      inputs[i][first_tensor_proto.name()] = q_tensors;
    }
  }
}

void InferenceExecutor::Init(const std::string &path, const std::string &dequant_path, size_t max_batch_size, bool prep_full_graphs) {
  // Init env
  if (InitEnv() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Server: InitEnv failed";
    return;
  }
  // Load full model as FuncGraph
  MindIRLoader model_loader;
  FuncGraphPtr full_func_graph = model_loader.LoadMindIR(path);
  if (full_func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Server: load MindIR model failed";
    return;
  }
  GraphId full_graph_id;
  KernelGraphPtr full_graph;
  Status ret = CompileGraph(full_func_graph, full_graph_id);
  if (ret != SUCCESS) {
    MS_LOG(EXCEPTION) << "Server: compile full graph failed";
    return;
  }

  full_graph = session_impl_->GetGraph(full_graph_id);
  if (full_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Server: GetGraph from session failed";
    return;
  }
  FuncGraphManagerPtr manager = MakeManager({full_graph});
  if (manager) {
    manager->AddFuncGraph(full_graph);
    full_graph->set_manager(manager);
  }
  // load profile res from proto file and construct cost_graph
  cost_graph_ = std::make_shared<CostGraph>(full_graph, path.substr(0, path.find_first_of('.')) + "_remote.prof");
  latency_graph_manager_.SetCostGraph(cost_graph_);
  latency_graph_manager_.SplitCostGraphIntoLatencyGraphs(is_dequant_, false);
  // add to context cache
  context_cache_.SetFuncGraph(full_func_graph);
  auto &cache_entry = context_cache_.AddContext(0, latency_graph_manager_.GetTotalLatencyGraphNums(), cost_graph_->GetSourceNode()->name_, "");
  std::vector<std::string> output_name = {"return"};
  cache_entry.AddGraph(1, full_graph_id, full_graph, output_name);
  cache_entry.GenerateInputVectors(full_graph);
  cache_entry.SetBaseTime(cost_graph_->GetFullLocalTime());
  context_cache_.SetFullGraphManager(1, manager);
  // generate full graphs for [2, max_batch_size] if needed
  if (prep_full_graphs) {
    std::vector<std::string> output_name = {"return"};
    for (size_t bsz = 2; bsz <= max_batch_size; ++bsz) {
      auto full_graph_res = context_cache_.GenerateFullKernelGraphByBatchSize(session_impl_, bsz);
      cache_entry.AddGraph(bsz, full_graph_res.first, full_graph_res.second, output_name);
    }
  }
  // init dequantizer
  if (is_dequant_) {
    dequantizer_ = std::unique_ptr<Dequantizer>(new Dequantizer(dequant_path, session_impl_));
  }
  // start run thread
  if (!is_running_.load()) {
    is_running_.store(true);
  }
  executor_ = std::thread(&InferenceExecutor::Run, this);
}

Status InferenceExecutor::InitEnv() {
  if (is_env_init_) {
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
  is_env_init_ = true;
  return SUCCESS;
}

Status InferenceExecutor::FinalizeEnv() {
  if (!is_env_init_) {
    MS_LOG(WARNING) << "Never initialize before.";
    return SUCCESS;
  }

  MS_LOG(INFO) << "Start finalize env";
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();

  is_env_init_ = false;
  MS_LOG(INFO) << "End finalize env";
  return SUCCESS;
}

Status InferenceExecutor::CompileGraph(const FuncGraphPtr &func_graph, GraphId &graph_id) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    graph_id = session_impl_->CompileGraph(NOT_NULL(func_graph));
    return SUCCESS;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "CompileGraph failed: " << e.what();
    return FAILED;
  }
}

void InferenceExecutor::Run() {
  while (is_running_.load()) {
    auto task = std::move(queue_.PopTask());
    // execute
    TaskPtr next_task;
    switch (task->plan->mode) {
      case NO_BATCHING:
        next_task = std::move(ExecuteTaskNoBatching(std::move(task)));
        break;
      case NAIVE_BATCHING:
        next_task = std::move(ExecuteTaskNaiveBatching(std::move(task)));
        break;
      case SNB_BATCHING:
        next_task = std::move(ExecuteTaskSNBBatching(std::move(task)));
        break;
      default:
        next_task = std::move(ExecuteTaskNoBatching(std::move(task)));
        break;
    }
    
    // form input or to complete queue
    if (is_last_stage_) {
      auto sched = sched_.lock();
      if (sched) {
        sched->EnqueueCompletionQueue(std::move(next_task));
      } else {
        MS_LOG(EXCEPTION) << "InferenceExecutor: Scheduler does not exist";
      }
    } else {
      // we expect that this branch should never be taken
      MS_LOG(EXCEPTION) << "InferenceExecutor: this StageExecutor should be the last stage";
    }
  }
}

void InferenceExecutor::Stop() {
  if (is_running_.load()) {
    is_running_.store(false);
    executor_.join();
    if (FinalizeEnv() != SUCCESS) {
      MS_LOG(ERROR) << "Clear env failed!";
    }
  }
}

void InferenceExecutor::EnqueueTask(TaskPtr&& task) {
  queue_.PushTask(std::move(task));
}

TaskPtr InferenceExecutor::ExecuteTaskNoBatching(TaskPtr&& task) {
  auto dequeue_time = TIMESTAMP();

  MS_EXCEPTION_IF_NULL(task);
  MS_EXCEPTION_IF_NULL(task->plan);
  // check input size and plan
  auto &plan = task->plan;
  if (plan->bsz_list.size() != 1 || 
      plan->bsz_list[0] != 1 || 
      task->inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "InferenceExecutor: execution in no batching mode requires exactly batch size = 1 with 1 request";
  }
  // dequant
  auto &inputs = task->inputs[0];
  NameToTensorMap input_map;
  for (auto &kv : inputs) {
    if (is_dequant_) {
      std::vector<ShapeVector> new_shapes_de = {kv.second[0]->shape(), kv.second[1]->shape(), kv.second[2]->shape()};
      auto dq_graph = dequantizer_->GetKernelGraphByShape(new_shapes_de);
      auto ret_d = dequantizer_->Dequantize(dq_graph, kv.second);
      if (ret_d.empty()) {
        MS_LOG(EXCEPTION) << "Server: dequantization failed";
      }
      input_map[kv.first] = ret_d[0];
    } else {
      input_map[kv.first] = kv.second[0];
    }
  }

  // find cut graph by entry point
  auto &entry_point = plan->entry_list[0];
  // fetch input parameter list from cache
  if (!context_cache_.FindContext(entry_point.second, "")) {
    auto &ctx = context_cache_.AddContext(entry_point.first, latency_graph_manager_.GetTotalLatencyGraphNums(), entry_point.second, "");
    std::vector<std::string> output_names;
    std::unordered_set<CostGraph::NodePtr> cut_nodes;
    cost_graph_->GetNodesByName(input_map, cut_nodes);
    auto cnode_list = latency_graph_manager_.GenerateKernelGraphSegmentServer(entry_point.first, cut_nodes);
    auto graph_id = cost_graph_->GenerateKernelGraphFromSegment(session_impl_, cost_graph_->GetFullKernelGraph(), cnode_list, output_names);
    auto graph = session_impl_->GetGraph(graph_id);
    ctx.AddGraph(1, graph_id, graph, output_names);
    ctx.GenerateInputVectors(graph);
  }

  auto &ctx = context_cache_.GetContext(entry_point.second, "");
  // emit task->input into the list
  auto &param_list = ctx.input_tensors;
  auto &emit_map = ctx.input_name_list_idx_map;
  for (auto &emit_it : emit_map) {
    if (input_map.count(emit_it.first) != 0) {
      param_list[emit_it.second] = input_map[emit_it.first];
    } else {
      MS_LOG(EXCEPTION) << "InferenceExecutor: cannot find input arg for Parameter named: " << emit_it.first;
    }
  }
  // call inference
  auto g = ctx.GetGraph(1); // assume we already have this graph in the cache
  VectorRef outputs;
  session_impl_->RunGraphAsync(g.first, param_list, &outputs);
  // cast return values
  auto ret = TransformVectorRefToMultiTensor(outputs);
  if (ret.empty()) {
    MS_LOG(EXCEPTION) << "InferenceExecutor: convert to Tensor failed, no output";
  }
  // assume there is only one batched tensor for each output
  auto next_task = std::unique_ptr<Task>(new Task());
  next_task->stage_id = 2;
  next_task->plan = plan;
  std::copy(task->session_ids.begin(), task->session_ids.end(), std::back_inserter(next_task->session_ids));
  next_task->inputs.resize(1);

  auto &ret_map = next_task->inputs[0];
  auto &output_names = ctx.output_names;
  for (size_t i = 0; i < ret.size(); ++i) {
    MS_EXCEPTION_IF_NULL(ret[i]);
    ret[i]->data_sync();
    ret_map[output_names[i]] = {ret[i]};
  }

  next_task->req_q_times.insert(next_task->req_q_times.end(), task->req_q_times.begin(), task->req_q_times.end());
  next_task->q_times[0] = task->q_times[0];
  next_task->e_times[0] = task->e_times[0];
  next_task->e_times[1] = task->e_times[1];
  // second stage queueing time
  next_task->q_times[1] = dequeue_time - task->q_times[1];
  auto exec_finish_time = TIMESTAMP();
  // second stage execution time
  next_task->e_times[2] = exec_finish_time - dequeue_time;
  next_task->q_times[2] = exec_finish_time;

  return next_task;
}

TaskPtr InferenceExecutor::ExecuteTaskNaiveBatching(TaskPtr&& task) {
  auto dequeue_time = TIMESTAMP();
  
  MS_EXCEPTION_IF_NULL(task);
  MS_EXCEPTION_IF_NULL(task->plan);
  // check input size and plan
  auto &plan = task->plan;
  if (plan->bsz_list.size() != 1 || 
      task->inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "InferenceExecutor: execution in naive batching mode requires batched requests at exactly 1 entry point";
  }
  // dequant
  auto &inputs = task->inputs[0];
  NameToTensorMap input_map;
  for (auto &kv : inputs) {
    if (is_dequant_) {
      std::vector<ShapeVector> new_shapes_de = {kv.second[0]->shape(), kv.second[1]->shape(), kv.second[2]->shape()};
      auto dq_graph = dequantizer_->GetKernelGraphByShape(new_shapes_de);
      auto ret_d = dequantizer_->Dequantize(dq_graph, kv.second);
      if (ret_d.empty()) {
        MS_LOG(EXCEPTION) << "Server: dequantization failed";
      }
      input_map[kv.first] = ret_d[0];
    } else {
      input_map[kv.first] = kv.second[0];
    }
  }
  // find cut graph by entry point
  auto batch_size = plan->bsz_list[0];
  auto &entry_point = plan->entry_list[0];
  if (!context_cache_.FindContext(entry_point.second, "")) {
    context_cache_.AddContext(entry_point.first, latency_graph_manager_.GetTotalLatencyGraphNums(), entry_point.second, "");
  }
  auto &ctx = context_cache_.GetContext(entry_point.second, "");
  // check if graph with current batch size exists
  if (!ctx.FindGraph(batch_size)) {
    if (entry_point.first != 0) {
      // check if full kernel graph with current batch size generated
      auto &full_graph_ctx = context_cache_.GetContext(cost_graph_->GetSourceNode()->name_, "");
      if (!full_graph_ctx.FindGraph(batch_size)) {
        std::vector<std::string> output_name = {"return"};
        auto full_graph_res = context_cache_.GenerateFullKernelGraphByBatchSize(session_impl_, batch_size);
        full_graph_ctx.AddGraph(batch_size, full_graph_res.first, full_graph_res.second, output_name);
      }
      // generate CNodeList
      std::vector<std::string> output_names;
      std::unordered_set<CostGraph::NodePtr> cut_nodes;
      cost_graph_->GetNodesByName(input_map, cut_nodes);
      auto cnode_list = latency_graph_manager_.GenerateKernelGraphSegmentServer(entry_point.first, cut_nodes);
      // translate cnode_list & generate cut graph
      auto full_graph = full_graph_ctx.GetGraph(batch_size);
      TranslateCNodeList(cost_graph_->GetFullKernelGraph(), full_graph.second, cnode_list);
      auto graph_id = cost_graph_->GenerateKernelGraphFromSegment(session_impl_, full_graph.second, cnode_list, output_names);
      auto graph = session_impl_->GetGraph(graph_id);
      ctx.AddGraph(batch_size, graph_id, graph, output_names);
      ctx.GenerateInputVectors(graph);
    } else {
      // generate full graph with batch size
      std::vector<std::string> output_name = {"return"};
      auto full_graph_res = context_cache_.GenerateFullKernelGraphByBatchSize(session_impl_, batch_size);
      ctx.AddGraph(batch_size, full_graph_res.first, full_graph_res.second, output_name);
    }
  }

  // fetch input parameter list from cache
  auto &param_list = ctx.input_tensors;
  auto &emit_map = ctx.input_name_list_idx_map;
  // emit task->input into the list
  for (auto &emit_it : emit_map) {
    if (input_map.count(emit_it.first) != 0) {
      param_list[emit_it.second] = input_map[emit_it.first];
    } else {
      MS_LOG(EXCEPTION) << "InferenceExecutor: cannot find input arg for Parameter named: " << emit_it.first;
    }
  }
  
  // call inference
  auto g = ctx.GetGraph(batch_size);
  VectorRef outputs;
  session_impl_->RunGraphAsync(g.first, param_list, &outputs);
  // cast return values
  auto ret = TransformVectorRefToMultiTensor(outputs);
  if (ret.empty()) {
    MS_LOG(EXCEPTION) << "InferenceExecutor: convert to Tensor failed, no output";
  }
  // assume there is only one batched tensor for each output
  auto next_task = std::unique_ptr<Task>(new Task());
  next_task->stage_id = 2;
  next_task->plan = plan;
  std::copy(task->session_ids.begin(), task->session_ids.end(), std::back_inserter(next_task->session_ids));
  next_task->inputs.resize(1);

  auto &ret_map = next_task->inputs[0];
  auto &output_names = ctx.output_names;
  for (size_t i = 0; i < ret.size(); ++i) {
    MS_EXCEPTION_IF_NULL(ret[i]);
    ret[i]->data_sync();
    ret_map[output_names[i]] = {ret[i]};
  }

  next_task->req_q_times.insert(next_task->req_q_times.end(), task->req_q_times.begin(), task->req_q_times.end());
  next_task->q_times[0] = task->q_times[0];
  next_task->e_times[0] = task->e_times[0];
  next_task->e_times[1] = task->e_times[1];
  // second stage queueing time
  next_task->q_times[1] = dequeue_time - task->q_times[1];
  auto exec_finish_time = TIMESTAMP();
  // second stage execution time
  next_task->e_times[2] = exec_finish_time - dequeue_time;
  next_task->q_times[2] = exec_finish_time;

  return next_task;
}

TaskPtr InferenceExecutor::ExecuteTaskSNBBatching(TaskPtr&& task) {
  auto dequeue_time = TIMESTAMP();

  MS_EXCEPTION_IF_NULL(task);
  MS_EXCEPTION_IF_NULL(task->plan);
  
  auto &plan = task->plan;
  auto stages = plan->entry_list.size();
  if (stages != task->inputs.size()) {
    MS_LOG(EXCEPTION) << "InferenceExecutor: execution in SNB batching mode requires the number of entry points equals to the number of input tensor maps";
  }

  // dequant
  std::vector<NameToTensorMap> input_maps;
  for (auto &inputs : task->inputs) {
    NameToTensorMap input_map;
    for (auto &kv : inputs) {
      if (is_dequant_) {
        std::vector<ShapeVector> new_shapes_de = {kv.second[0]->shape(), kv.second[1]->shape(), kv.second[2]->shape()};
        auto dq_graph = dequantizer_->GetKernelGraphByShape(new_shapes_de);
        auto ret_d = dequantizer_->Dequantize(dq_graph, kv.second);
        if (ret_d.empty()) {
          MS_LOG(EXCEPTION) << "Server: dequantization failed";
        }
        input_map[kv.first] = ret_d[0];
      } else {
        input_map[kv.first] = kv.second[0];
        LoadIntermidiateData(kv.second[0]);
      }
    }
    input_maps.emplace_back(input_map);
  }

  auto next_task = std::unique_ptr<Task>(new Task());
  next_task->stage_id = 2;
  next_task->plan = plan;
  std::copy(task->session_ids.begin(), task->session_ids.end(), std::back_inserter(next_task->session_ids));
  next_task->inputs.resize(1);

  size_t cur_batch_size = 0;
  double total_base_time = 0.0;
  for (size_t stage = 0; stage < stages; ++stage) {
    // find cut graph by entry point
    auto batch_size = plan->bsz_list[stage];
    cur_batch_size += batch_size;
    const auto &entry_point = plan->entry_list[stage];
    const auto &exit_point = (stage == stages - 1) ? std::make_pair(latency_graph_manager_.GetTotalLatencyGraphNums(), "") : plan->entry_list[stage + 1];
    // find cut graph by entry point and exit point
    if (!context_cache_.FindContext(entry_point.second, exit_point.second)) {
      context_cache_.AddContext(entry_point.first, exit_point.first, entry_point.second, exit_point.second);
    }
    auto &ctx = context_cache_.GetContext(entry_point.second, exit_point.second);
    // check if graph with current batch size exisits
    if (!ctx.FindGraph(cur_batch_size)) {
      if (entry_point.first != 0 || exit_point.second != "") {
        // check if full kernel graph with current batch size generated
        auto &full_graph_ctx = context_cache_.GetContext(cost_graph_->GetSourceNode()->name_, "");
        if (!full_graph_ctx.FindGraph(cur_batch_size)) {
          std::vector<std::string> output_name = {"return"};
          auto full_graph_res = context_cache_.GenerateFullKernelGraphByBatchSize(session_impl_, cur_batch_size);
          full_graph_ctx.AddGraph(cur_batch_size, full_graph_res.first, full_graph_res.second, output_name);
        }
        // generate CNodeList
        std::vector<std::string> output_names;
        std::unordered_set<CostGraph::NodePtr> cut_nodes_s;
        std::unordered_set<CostGraph::NodePtr> cut_nodes_e;
        cost_graph_->GetNodesByName(input_maps[stage], cut_nodes_s);
        if (stage != stages - 1) {
          cost_graph_->GetNodesByName(input_maps[stage + 1], cut_nodes_e);
        }

        double base_time = 0.0;
        auto cnode_list = latency_graph_manager_.GenerateKernelGraphSegmentBetween(entry_point.first, cut_nodes_s, exit_point.first, cut_nodes_e, base_time);
        // translate cnode_list & generate cut graph
        auto full_graph = full_graph_ctx.GetGraph(cur_batch_size);
        TranslateCNodeList(cost_graph_->GetFullKernelGraph(), full_graph.second, cnode_list);
        auto graph_id = cost_graph_->GenerateKernelGraphFromSegment(session_impl_, full_graph.second, cnode_list, output_names);
        auto graph = session_impl_->GetGraph(graph_id);
        ctx.AddGraph(cur_batch_size, graph_id, graph, output_names);
        ctx.GenerateInputVectors(graph);
        ctx.SetBaseTime(base_time);
      } else {
        // generate full graph with batch size
        std::vector<std::string> output_name = {"return"};
        auto full_graph_res = context_cache_.GenerateFullKernelGraphByBatchSize(session_impl_, cur_batch_size);
        ctx.AddGraph(cur_batch_size, full_graph_res.first, full_graph_res.second, output_name);
      }
    }

    auto &param_list = ctx.input_tensors;
    auto &emit_map = ctx.input_name_list_idx_map;
    // emit task->input into the list
    auto &input_map = input_maps[stage];
    for (auto &emit_it : emit_map) {
      if (input_map.count(emit_it.first) != 0) {
        param_list[emit_it.second] = input_map[emit_it.first];
      } else {
        MS_LOG(EXCEPTION) << "InferenceExecutor: cannot find input arg for Parameter named: " << emit_it.first;
      }
    }

    // call inference
    auto g = ctx.GetGraph(cur_batch_size);
    total_base_time += ctx.GetBaseTime();
    VectorRef outputs;
    session_impl_->RunGraphAsync(g.first, param_list, &outputs);
    // cast return values
    auto ret = TransformVectorRefToMultiTensor(outputs);
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "InferenceExecutor: convert to Tensor failed, no output";
    }

    // async memcpy when not at the last stage / wait for final result at the last stage
    auto &output_names = ctx.output_names;
    if (stage != stages - 1) {
      auto &next_input_map = input_maps[stage + 1];
      for (size_t i = 0; i < ret.size(); ++i) {
        // notice that next_stage_inputs should all reside in GPU
        auto next_stage_input_it = next_input_map.find(output_names[i]);
        if (next_stage_input_it == next_input_map.end()) {
          MS_LOG(EXCEPTION) << "InferenceExecutor: cannot find input named: " << output_names[i] << " of graph with entry point " << exit_point.second;
        }
        auto merged_input = MergeTensorPair(ret[i], next_stage_input_it->second);
        MS_EXCEPTION_IF_NULL(merged_input);
        next_stage_input_it->second = merged_input;
      }
    } else {
      auto &ret_map = next_task->inputs[0];
      auto &output_names = ctx.output_names;
      for (size_t i = 0; i < ret.size(); ++i) {
        MS_EXCEPTION_IF_NULL(ret[i]);
        ret[i]->data_sync();
        ret_map[output_names[i]] = {ret[i]};
      }
    }
  }

  next_task->req_q_times.insert(next_task->req_q_times.end(), task->req_q_times.begin(), task->req_q_times.end());
  next_task->q_times[0] = task->q_times[0];
  next_task->e_times[0] = task->e_times[0];
  next_task->e_times[1] = task->e_times[1];
  // second stage queueing time
  next_task->q_times[1] = dequeue_time - task->q_times[1];
  auto exec_finish_time = TIMESTAMP();
  // second stage execution time
  next_task->e_times[2] = exec_finish_time - dequeue_time;
  next_task->q_times[2] = exec_finish_time;
  next_task->factor = (double)next_task->e_times[2] / total_base_time;

  return next_task;
}

}
}
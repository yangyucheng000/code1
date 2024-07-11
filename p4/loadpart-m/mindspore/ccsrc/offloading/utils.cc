#include "utils.h"

namespace mindspore {

namespace offloading {

static std::unordered_map<int, offloading_serving::TensorProto_DataType> g_data_type_map = {
  {kNumberTypeBool, offloading_serving::TensorProto_DataType_BOOL},
  {kNumberTypeInt8, offloading_serving::TensorProto_DataType_INT8},
  {kNumberTypeInt16, offloading_serving::TensorProto_DataType_INT16},
  {kNumberTypeInt32, offloading_serving::TensorProto_DataType_INT32},
  {kNumberTypeInt64, offloading_serving::TensorProto_DataType_INT64},
  {kNumberTypeUInt8, offloading_serving::TensorProto_DataType_UINT8},
  {kNumberTypeUInt16, offloading_serving::TensorProto_DataType_UINT16},
  {kNumberTypeUInt32, offloading_serving::TensorProto_DataType_UINT32},
  {kNumberTypeUInt64, offloading_serving::TensorProto_DataType_UINT64},
  {kNumberTypeFloat16, offloading_serving::TensorProto_DataType_FLOAT16},
  {kNumberTypeFloat32, offloading_serving::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat64, offloading_serving::TensorProto_DataType_DOUBLE},
  {kObjectTypeString, offloading_serving::TensorProto_DataType_STRING},
};    

static std::unordered_map<int, TypeId> kDefaultValueSwitchMap{
  {offloading_serving::TensorProto_DataType_BOOL, kNumberTypeBool},
  {offloading_serving::TensorProto_DataType_INT8, kNumberTypeInt8},
  {offloading_serving::TensorProto_DataType_INT16, kNumberTypeInt16},
  {offloading_serving::TensorProto_DataType_INT32, kNumberTypeInt32},
  {offloading_serving::TensorProto_DataType_INT64, kNumberTypeInt64},
  {offloading_serving::TensorProto_DataType_UINT8, kNumberTypeUInt8},
  {offloading_serving::TensorProto_DataType_UINT16, kNumberTypeUInt16},
  {offloading_serving::TensorProto_DataType_UINT32, kNumberTypeUInt32},
  {offloading_serving::TensorProto_DataType_UINT64, kNumberTypeUInt64},
  {offloading_serving::TensorProto_DataType_FLOAT16, kNumberTypeFloat16},
  {offloading_serving::TensorProto_DataType_FLOAT, kNumberTypeFloat32},
  {offloading_serving::TensorProto_DataType_FLOAT64, kNumberTypeFloat64},
  {offloading_serving::TensorProto_DataType_DOUBLE, kNumberTypeFloat64},
  {offloading_serving::TensorProto_DataType_STRING, kObjectTypeString},
};

bool OffloadingContextCache::FindContext(const std::string &cut_nodes_string) {
  return (cache_.find(cut_nodes_string) != cache_.end());
}

void OffloadingContextCache::AddContext(const size_t lg_idx, const std::string &cut_nodes_string, const GraphId gid, KernelGraphPtr g, std::vector<std::string> &output_names) {
  auto ctx = OffloadingContext(gid, g);
  ctx.output_names = output_names;
  cache_[cut_nodes_string] = ctx;
  cut_nodes_to_lg_idx_[cut_nodes_string] = lg_idx;
}

OffloadingContext& OffloadingContextCache::GetContext(const std::string &cut_nodes_string) {
  auto it = cache_.find(cut_nodes_string);
  if (it == cache_.end()) {
    MS_LOG(EXCEPTION) << "OffloadingContextCache: cut nodes " << cut_nodes_string << " not found";
  }
  return it->second;
}

BatchOffloadingContext::BatchOffloadingContext(size_t batch_size, const GraphId gid, KernelGraphPtr g) {
  graphs[batch_size] = std::make_pair(gid, g);
}

bool BatchOffloadingContext::FindGraph(size_t batch_size) {
  return (graphs.find(batch_size) != graphs.end());
}

void BatchOffloadingContext::AddGraph(size_t batch_size, const GraphId gid, KernelGraphPtr g, std::vector<std::string> &o_names) {
  graphs[batch_size] = std::make_pair(gid, g);
  if (output_names.empty()) {
    output_names = o_names;
  }
}

std::pair<GraphId, KernelGraphPtr>& BatchOffloadingContext::GetGraph(size_t batch_size) {
  auto it = graphs.find(batch_size);
  if (it == graphs.end()) {
    MS_LOG(EXCEPTION) << "BatchOffloadingContext: graph with batch size = " << batch_size << " not found";
  }
  return it->second;
}

void BatchOffloadingContext::GenerateInputVectors(KernelGraphPtr &graph) {
  if (!input_tensors.empty()) return;
  MS_EXCEPTION_IF_NULL(graph);
  auto& input_nodes = graph->inputs();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto input_param_ptr = (input_nodes[i])->cast<ParameterPtr>();
    if (input_param_ptr->has_default()) {
      if (!input_param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << input_param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      auto input_tensor_ptr = input_param_ptr->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(input_tensor_ptr);
      input_tensors.push_back(input_tensor_ptr);
    } else {
      auto name = input_param_ptr->DebugString(0);
      input_tensors.push_back(nullptr);
      input_name_list_idx_map[name] = i;
    }
  }
}

bool BatchOffloadingContextCache::FindContext(const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string) {
  auto start_it = cache_.find(start_cut_nodes_string);
  if (start_it != cache_.end()) {
    auto end_it = start_it->second.find(end_cut_nodes_string);
    if (end_it != start_it->second.end()) {
      return true;
    }
  }
  return false;
}

BatchOffloadingContext& BatchOffloadingContextCache::AddContext(const size_t start_lg_idx, const size_t end_lg_idx, const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string) {
  auto start_it = cache_.find(start_cut_nodes_string);
  if (start_it == cache_.end()) {
    auto ret = cache_.emplace(std::make_pair(start_cut_nodes_string, std::unordered_map<std::string, BatchOffloadingContext>()));
    start_it = ret.first;
  }

  auto end_it = start_it->second.find(end_cut_nodes_string);
  if (end_it == start_it->second.end()) {
    auto ret = start_it->second.emplace(std::make_pair(end_cut_nodes_string, BatchOffloadingContext()));
    end_it = ret.first;
  }

  cut_nodes_to_lg_idx_[start_cut_nodes_string] = start_lg_idx;
  cut_nodes_to_lg_idx_[end_cut_nodes_string] = end_lg_idx;

  return end_it->second;
}

BatchOffloadingContext& BatchOffloadingContextCache::GetContext(const std::string &start_cut_nodes_string, const std::string &end_cut_nodes_string) {
  auto start_it = cache_.find(start_cut_nodes_string);
  if (start_it != cache_.end()) {
    auto end_it = start_it->second.find(end_cut_nodes_string);
    if (end_it != start_it->second.end()) {
      return end_it->second;
    }
  }
  MS_LOG(EXCEPTION) << "BatchOffloadingContextCache: cut nodes [" << start_cut_nodes_string << ", " << end_cut_nodes_string << "] not found";
}

std::pair<GraphId, KernelGraphPtr> BatchOffloadingContextCache::GenerateFullKernelGraphByBatchSize(session::SessionPtr session_impl, size_t batch_size) {
  if (batch_size != 1) {
    const auto &inputs = full_func_graph_->get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto &param = inputs[i];
      auto shape_ptr = std::dynamic_pointer_cast<abstract::Shape>(param->Shape());
      if (shape_ptr == nullptr) {
        MS_LOG(ERROR) << "inputs " << i << " is not supported to resize, debug string: " << param->DebugString();
      }
      auto tmp_shape = shape_ptr->shape();
      tmp_shape[0] = batch_size;
      shape_ptr->set_shape(tmp_shape);
    }
  }
  // compile & get graph
  GraphId full_graph_id;
  KernelGraphPtr full_graph;
  try {
    if (batch_size != 1) {
      full_graph_id = session_impl->CompileGraphWithInferShape(NOT_NULL(full_func_graph_));
    } else {
      full_graph_id = session_impl->CompileGraph(NOT_NULL(full_func_graph_));
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Server: compile full graph failed: " << e.what();
  }

  full_graph = session_impl->GetGraph(full_graph_id);
  if (full_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Server: GetGraph from session failed";
  }
  FuncGraphManagerPtr manager = MakeManager({full_graph});
  if (manager) {
    manager->AddFuncGraph(full_graph);
    full_graph->set_manager(manager);
  }
  SetFullGraphManager(batch_size, manager);
  return std::make_pair(full_graph_id, full_graph);
}

void FixedSizeBuffer::PopFront() {
  if (buf_.empty()) {
    return;
  }
  buf_.pop_front();
}

void FixedSizeBuffer::PopBack() {
  if (buf_.empty()) {
    return;
  }
  buf_.pop_back();
}

void FixedSizeBuffer::Push(const double x) {
  if (buf_.size() == max_size_) {
    buf_.pop_front();
  }
  buf_.push_back(x);
}

double FixedSizeBuffer::GetAvgValue() {
  auto sum = std::accumulate(buf_.begin(), buf_.end(), 0.0);
  return sum / buf_.size();
}

bool FixedSizeBuffer::IsEmpty() {
  return buf_.empty();
}

void FixedSizeBuffer::Clear() {
  buf_.clear();
}


TypeId GetTypeIdFromProtoTensor(const offloading_serving::TensorProto &tensor_proto) {
  if (kDefaultValueSwitchMap.find(tensor_proto.data_type()) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "offloading_serving TensorProto data_type is not support yet!";
  }
  return kDefaultValueSwitchMap[tensor_proto.data_type()];
}

bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  return (prim && prim_set.find(prim) != prim_set.end());
}

void UpdateKernelArgs(CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  std::string device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
#if ENABLE_GPU
  if (device_target == kCPUDevice) {
    auto cpu_kernel_mod = dynamic_cast<kernel::CPUKernel *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(cpu_kernel_mod);
    cpu_kernel_mod->Init(cnode);
  } else if (device_target == kGPUDevice) {
    auto gpu_kernel_mod = dynamic_cast<kernel::GpuKernel *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(gpu_kernel_mod);
    gpu_kernel_mod->DestroyResource();
    gpu_kernel_mod->ResetResource();
    gpu_kernel_mod->Init(cnode);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported device target";
    return;
  }
#elif ENABLE_CPU
  if (device_target == kCPUDevice) {
    auto cpu_kernel_mod = dynamic_cast<kernel::CPUKernel *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(cpu_kernel_mod);
    cpu_kernel_mod->Init(cnode);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported device target";
    return;
  }
#endif
}

bool CheckAllTensor(const ValueTuplePtr &value_tuple) {
  auto elements = value_tuple->value();
  for (auto element : elements) {
    if (!(element->isa<ValueTuple>() && CheckAllTensor(element->cast<ValueTuplePtr>())) &&
        !(element->isa<tensor::MetaTensor>())) {
      return false;
    }
  }
  return true;
}

AbstractBasePtr ValueToAbstract(const ValuePtr &value, bool enable_tuple_broaden) {
  MS_EXCEPTION_IF_NULL(value);
  bool broaden = value->isa<tensor::MetaTensor>() ||
                 (enable_tuple_broaden && value->isa<ValueTuple>() && CheckAllTensor(value->cast<ValueTuplePtr>())) ||
                 (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && value->isa<Scalar>());

  return abstract::FromValue(value, broaden);
}

offloading_serving::TensorProto_DataType GetProtoDataType(TypeId type_id) {
  auto iter = g_data_type_map.find(type_id);
  if (iter == g_data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Convert type error, unsupported type! " << type_id;
  }
  return iter->second;
}

void TensorToProtoTensor(const tensor::TensorPtr &tensor, offloading_serving::TensorProto *const tensor_proto) {
  if(tensor == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "TensorPtr or TensorProto is null!";
    return;
  }
  auto dtype = tensor->data_type();
  const auto &dims = tensor->shape();
  tensor_proto->set_data_type(GetProtoDataType(dtype));
  for (const auto &dim : dims) {
    tensor_proto->add_dims(dim);
  }
  tensor_proto->set_name(tensor->id()); // need to specify intermidiate result is the output of which CNode
  tensor_proto->set_raw_data(tensor->data_c(), tensor->data().nbytes());
  tensor_proto->set_compressed(false);
}

tensor::TensorPtr ProtoTensorToTensor(const offloading_serving::TensorProto &tensor_proto) {
  ShapeVector shape;
  for (int i = 0; i < tensor_proto.dims_size(); ++i) {
    shape.push_back(tensor_proto.dims(i));
  }

  auto dtype = GetTypeIdFromProtoTensor(tensor_proto);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(dtype, shape);
  const std::string &init_data = tensor_proto.raw_data();
  auto *data_buf = reinterpret_cast<uint8_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(data_buf);
  auto ret = memcpy_s(data_buf, tensor->data().nbytes(), init_data.data(), init_data.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error for building Tensor from TensorProto, errorno " << ret;
    return nullptr;
  }
  return tensor;
}

py::object TensorToPyData(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->NeedWait()) {
    py::gil_scoped_release release;
    tensor->Wait();
  }
  py::tuple v(1);
  v[0] = tensor;
  return v[0];
}

tensor::TensorPtr PyDataToTensor(const py::object &obj) {
  if (py::isinstance<py::array>(obj)) {
    MS_LOG(ERROR) << "obj is a numpy array, not a tensor.";
    return nullptr;
  }
  ValuePtr converted = nullptr;
  bool succ = parse::ConvertData(obj, &converted);
  if (!succ) {
    MS_LOG(ERROR) << "obj convertion failed.";
    return nullptr;
  }
  if (converted->isa<tensor::Tensor>()) {
    return converted->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(ERROR) << "obj cannot be converted to Tensor.";
    return nullptr;
  }
}

bool CheckInputArgShape(const ValuePtr &value, const ParameterPtr &input_node) {
  if (input_node->has_default()) {
    MS_LOG(EXCEPTION) << "Redundant input";
    return false;
  }

  auto single_shape_checker = [](const tensor::TensorPtr &tensor, const abstract::BaseShapePtr &shape) {
    const auto &param_dims = shape->cast<abstract::ShapePtr>()->shape();
    const auto &tensor_dims = tensor->shape();
    if (param_dims.size() != tensor_dims.size()) {
      MS_LOG(ERROR) << "Shape error: incompatible dimensions";
      return false;
    }
    for (size_t i = 0; i < param_dims.size(); ++i) {
      if (param_dims[i] != tensor_dims[i]) {
        MS_LOG(ERROR) << "Shape error: incompatible at dimension " << i;
        return false;
      }
    }
    return true;
  };

  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto shape = input_node->Shape();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(ERROR) << "Shape error: Tensor and Parameter shape type are incompatible";
      return false;
    }
    return single_shape_checker(tensor, shape);
  } else if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    auto shape = input_node->Shape();
    if (!shape->isa<abstract::TupleShape>()) {
      MS_LOG(ERROR) << "Shape error: ValueTuple and Parameter shape type are incompatible";
      return false;
    }
    auto tuple_shape = shape->cast<abstract::TupleShapePtr>();
    if (tuple_shape->size() != value_tuple->size()) {
      MS_LOG(ERROR) << "Shape error: ValueTuple size and Parameter shape size are incompatible";
      return false;
    }
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        auto tensor_shape = (*tuple_shape)[i];
        if (!tensor_shape->isa<abstract::Shape>()) {
          MS_LOG(ERROR) << "Shape error: Tensor shape should be in Shape type";
          return false;
        }
        if (!single_shape_checker(tensor, tensor_shape)) {
          MS_LOG(ERROR) << "Shape error: ValueTuple shape incompatible at " << i;
          return false;
        }
      } else {
        MS_LOG(ERROR) << "Shape error: ValueTuple should contain only Tensors";
        return false;
      }
    }
  } else {
    MS_LOG(ERROR) << "Shape error: Value is neither Tensor nor ValueTuple";
    return false;
  }
  return true;
}

void GetOrderedCnodesVector(FuncGraphPtr &graph, CNodePtrList &cnodes) {
  auto BelongSameGraph = std::bind(IncludeBelongGraph, graph, std::placeholders::_1);
  auto SuccDepends = std::bind(SuccIncludeFV, graph, std::placeholders::_1);
  auto nodes = TopoSort(graph->get_return(), SuccDepends, BelongSameGraph);
  for (const auto &node : nodes) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode) {
      cnodes.push_back(cnode);
    }
  }
}

bool CompareInput(const tensor::TensorPtr &input, const ParameterPtr &parameter) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(parameter);
  // compare dims
  auto parameter_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);

  // compare shape
  auto input_shape = input->shape();
  vector<size_t> trans_input;
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(trans_input),
                       [](const int64_t dim) { return static_cast<size_t>(dim); });
  auto is_scalar_shape = [](const vector<size_t> &shape) {
    return shape.empty() || (shape.size() == 1 && shape[0] == 1);
  };
  if ((!is_scalar_shape(trans_input) || !is_scalar_shape(parameter_shape)) && (trans_input != parameter_shape)) {
    MS_LOG(ERROR) << "Input shape is inconsistent.";
    return false;
  }

  // compare data type
  auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
  if (input->data_type() != kernel_build_info->GetOutputDeviceType(0)) {
    MS_LOG(ERROR) << "Input data type is inconsistent.";
    return false;
  }
  return true;
}

void PrintParameterShape(const ParameterPtr &parameter) {
  MS_EXCEPTION_IF_NULL(parameter);
  auto parameter_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
  std::cout << parameter->fullname_with_scope() << ": [";
  for (size_t i = 0; i < parameter_shape.size(); ++i) {
    std::cout << parameter_shape[i];
    if (i == parameter_shape.size() - 1) {
      break;
    }
    std::cout << ", ";
  }
  std::cout << "]" << std::endl;
}

void GetSegmentOutput(KernelGraphPtr &origin_graph, const CNodePtrList &node_list, AnfNodePtrList &output_nodes) {
  std::unordered_set<AnfNodePtr> node_set(node_list.begin(), node_list.end());
  auto manager = origin_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto& node_user_map = manager->node_users();
  output_nodes.clear();
  for (auto &n : node_list) {
    auto it = node_user_map.find(n);
    if (it == node_user_map.end()) {
      MS_LOG(EXCEPTION) << "CNode " << n->fullname_with_scope() << " is not a node of the full graph";
      output_nodes.clear();
      return;
    }
    // some Load has two user nodes
    if (IsPrimitiveCNode(n, prim::kPrimLoad)) {
      auto &output_info_list = it->second;
      for (auto &info : output_info_list) {
        if (IsPrimitiveCNode(info.first, prim::kPrimMakeTuple) || IsPrimitiveCNode(info.first, prim::kPrimUpdateState)) {
          // these two nodes never exist in the input CNode list
          continue;
        }
        if (node_set.find(info.first) == node_set.end()) {
          MS_LOG(EXCEPTION) << "User node of Load CNode " << n->fullname_with_scope() << " does not exist in the input CNode list";
          output_nodes.clear();
          return;
        }
      }
      continue;
    }
    auto &output_info_list = it->second;
    for (auto &info : output_info_list) {
      // may never true
      if ((IsPrimitiveCNode(info.first, prim::kPrimDepend) && info.second == kDependAttachNodeIndex) ||
          (IsPrimitiveCNode(info.first, prim::kPrimUpdateState))) {
        continue;
      }
      if (node_set.find(info.first) == node_set.end()) {
        // CNode of other graph
        if (IsPrimitiveCNode(n, prim::kPrimMakeTuple)) {
          MS_LOG(EXCEPTION) << "MakeTuple node " << n->fullname_with_scope() << " cannot be used by a CNode in other graph";
          output_nodes.clear();
          return;
        }
        output_nodes.push_back(n);
        break;
      }
    }
  }
}

void PrintTensorData(tensor::TensorPtr &tensor) {
  auto type_id = tensor->data_type();
  if (type_id != kNumberTypeFloat32 && type_id != kNumberTypeFloat) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
    return;
  }
  auto data_ptr = static_cast<float*>(tensor->data_c());
  auto elem_num = tensor->DataSize();
  for (size_t i = 0; i < elem_num; ++i) {
    std::cout << data_ptr[i] << ' ';
  }
  std::cout << std::endl;
}

AnfNodePtr TranslateAnfNode(KernelGraphPtr &from_g, KernelGraphPtr &to_g, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(from_g);
  MS_EXCEPTION_IF_NULL(to_g);
  if (from_g == to_g) return node;
  auto &bf_map_from = from_g->backend_front_anf_map();
  auto &fb_map_to = to_g->front_backend_anf_map();
  return fb_map_to.at(bf_map_from.at(node));
}

void TranslateCNodeList(KernelGraphPtr &from_g, KernelGraphPtr &to_g, CNodePtrList &cnode_list) {
  MS_EXCEPTION_IF_NULL(from_g);
  MS_EXCEPTION_IF_NULL(to_g);
  if (from_g == to_g) return;
  auto &bf_map_from = from_g->backend_front_anf_map();
  auto &fb_map_to = to_g->front_backend_anf_map();
  std::transform(cnode_list.cbegin(), cnode_list.cend(), cnode_list.begin(), [&](const CNodePtr &node) {
    return fb_map_to.at(bf_map_from.at(node->cast<AnfNodePtr>()))->cast<CNodePtr>();
  });
}

}
}
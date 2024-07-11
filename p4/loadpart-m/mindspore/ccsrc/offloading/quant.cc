#include "quant.h"

namespace mindspore {

namespace offloading {

static std::set<std::string> compressors = {"blosclz", "lz4", "lz4hc", "zlib", "zstd"};

std::string GenerateShapeKey(const std::vector<ShapeVector> &shapes) {
  std::string shape_key;
  for (size_t i = 0; i < shapes.size(); ++i) {
    shape_key += std::to_string(i) + ":";
    for (size_t j = 0; j < shapes[i].size(); ++j) {
      shape_key += std::to_string(shapes[i][j]);
      if (j + 1 < shapes[i].size()) {
        shape_key += ",";
      }
    }
    if (i + 1 < shapes.size()) {
      shape_key += ";";
    }
  }
  return shape_key;
}

Compressor::Compressor(int clevel, int do_shuffle, int nthreads, std::string compressor)
  : clevel_(clevel), do_shuffle_(do_shuffle), nthreads_(nthreads), compressor_(compressor), buffer_size_(COMPRESS_BUFFER_SIZE)
{
  if (compressors.find(compressor) == compressors.end()) {
    MS_LOG(EXCEPTION) << "Compressor: unknown compressor type " << compressor;
  }
  blosc2_init();
  blosc2_set_nthreads(nthreads_);
  blosc1_set_compressor(compressor_.c_str());
  buffer_ = malloc(buffer_size_);
  MS_EXCEPTION_IF_NULL(buffer_);
  // for logging 
  log_file.open("compressor.log");
  if (!log_file.is_open()) {
    MS_LOG(EXCEPTION) << "Open file compressor.log failed!";
  }
}

size_t Compressor::Compress(const void *src, size_t typesize, size_t nbytes) {
  // check buffer size
  auto start_time = TIMESTAMP();
  if (buffer_size_ < nbytes) {
    // not thread-safe when pipelining
    free(buffer_);
    while (buffer_size_ < nbytes) buffer_size_ *= 2;
    buffer_ = malloc(buffer_size_);
    MS_EXCEPTION_IF_NULL(buffer_);
  }
  auto csize = blosc2_compress(clevel_, do_shuffle_, typesize, src, nbytes, buffer_, buffer_size_);
  if (csize < 0) {
    free(buffer_);
    MS_LOG(EXCEPTION) << "Compressor: compress failed with error code " << csize;
    return csize;
  }
  // log_file << nbytes * 4 << " " << nbytes << " " << csize << "\n";
  auto end_time = TIMESTAMP();
  log_file << nbytes << '\t' << csize << '\t' << end_time - start_time << "\n";
  return csize;
}

void* Compressor::GetBuffer() {
  MS_EXCEPTION_IF_NULL(buffer_);
  return buffer_;
}

Compressor::~Compressor() {
  blosc2_destroy();
  free(buffer_);
  // for logging
  log_file.close();
}

Decompressor::Decompressor(int nthreads)
  : nthreads_(nthreads) 
{
  blosc2_init();
  blosc2_set_nthreads(nthreads_);
}

size_t Decompressor::Decompress(const void *src, size_t nbytes, void *dest, size_t destsize) {
  auto dsize = blosc2_decompress(src, nbytes, dest, destsize);
  if (dsize < 0) {
    MS_LOG(EXCEPTION) << "Decompressor: decompress failed with error code " << dsize;
    return dsize;
  }
  return dsize;
}

Decompressor::~Decompressor() {
  blosc2_destroy();
}

Quantizer::Quantizer(const std::string &path, session::SessionPtr session_impl) {
  session_impl_ = session_impl;
  if (Load(path) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Quantizer: Load quantize computation graph failed";
  }
}

Status Quantizer::Load(const std::string &path) {
  MindIRLoader model_loader;
  for (size_t i = 2; i <= 4; ++i) {
    auto subpath = path + "_" + std::to_string(i) + ".mindir";
    auto model = model_loader.LoadMindIR(subpath);
    if (model == nullptr) {
      MS_LOG(EXCEPTION) << "Quantizer: load MindIR model failed";
      return FAILED;
    }
    auto model_manager = MakeManager({model});
    if (model_manager) {
      model_manager->AddFuncGraph(model);
      model->set_manager(model_manager);
    }
    models_[i] = model;
    model_managers_[i] = model_manager;
  }
  return SUCCESS;
}

KernelGraphPtr Quantizer::Compile(size_t dim) {
  MS_EXCEPTION_IF_NULL(session_impl_);
  auto &model = models_.at(dim);
  try {
    auto graph_id = session_impl_->CompileGraphWithInferShape(NOT_NULL(model));
    // auto graph_id = session_impl_->CompileGraph(NOT_NULL(model_));
    auto graph = session_impl_->GetGraph(graph_id);
    return graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Quantizer: CompileGraph failed: " << e.what();
    return nullptr;
  }
}

Status Quantizer::Resize(std::vector<ShapeVector> &shapes) {
  std::string shape_key = GenerateShapeKey(shapes);
  return Resize(shapes, shape_key);
}

Status Quantizer::Resize(std::vector<ShapeVector> &shapes, const std::string &shape_key) {
  if (auto it = graph_cache_.find(shape_key); it != graph_cache_.end()) {
    MS_LOG(WARNING) << "Quantizer: KernelGraph for shape " << shape_key << " has been generated";
    return SUCCESS; 
  }
  auto dim = shapes[0].size();
  auto &model = models_.at(dim);
  const auto &inputs = model->parameters();
  // const auto &node_user_map = model_->manager()->node_users();
  if (inputs.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "Quantizer: shapes size " << shapes.size() << " does not match with parameter numbers " << inputs.size();
    return FAILED;
  }
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto &param = inputs[i];
    auto shape_ptr = std::dynamic_pointer_cast<abstract::Shape>(param->Shape());
    if (shape_ptr == nullptr) {
      MS_LOG(ERROR) << "Quantizer: inputs " << i << " is not supported to resize, debug string: " << param->DebugString();
      return FAILED;
    }
    shape_ptr->set_shape(shapes[i]);
  }

  auto graph = Compile(dim);
  if (graph == nullptr) {
    MS_LOG(EXCEPTION) << "Quantizer: Compile failed for shape " << shape_key;
    return FAILED;
  }
  graph_cache_[shape_key] = graph;
  
  return SUCCESS;
}

tensor::TensorPtrList Quantizer::Quantize(KernelGraphPtr graph, const tensor::TensorPtrList &inputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Quantizer: input number should be exactly 1";
    return {};
  }
  VectorRef outputs;
  session_impl_->RunGraphAsync(graph->graph_id(), inputs, &outputs);
  auto ret = TransformVectorRefToMultiTensor(outputs);
  return ret;
}

KernelGraphPtr Quantizer::GetKernelGraphByShape(std::vector<ShapeVector> &shapes) {
  std::string shape_key = GenerateShapeKey(shapes);
  auto it = graph_cache_.find(shape_key);
  if (it == graph_cache_.end()) {
    Resize(shapes);
    return graph_cache_.find(shape_key)->second; 
  }
  return it->second;
}

Dequantizer::Dequantizer(const std::string &path, session::SessionPtr session_impl) {
  session_impl_ = session_impl;
  if (Load(path) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Dequantizer: Load quantize computation graph failed";
  }
}

Status Dequantizer::Load(const std::string &path) {
  MindIRLoader model_loader;
  model_ = model_loader.LoadMindIR(path);
  if (model_ == nullptr) {
    MS_LOG(EXCEPTION) << "Dequantizer: load MindIR model failed";
    return FAILED;
  }
  model_manager_ = MakeManager({model_});
  if (model_manager_) {
    model_manager_->AddFuncGraph(model_);
    model_->set_manager(model_manager_);
  }
  return SUCCESS;
}

KernelGraphPtr Dequantizer::Compile() {
  MS_EXCEPTION_IF_NULL(session_impl_);
  MS_EXCEPTION_IF_NULL(model_);
  try {
    auto graph_id = session_impl_->CompileGraphWithInferShape(NOT_NULL(model_));
    // auto graph_id = session_impl_->CompileGraph(NOT_NULL(model_));
    auto graph = session_impl_->GetGraph(graph_id);
    return graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Dequantizer: CompileGraph failed: " << e.what();
    return nullptr;
  }
}

Status Dequantizer::Resize(std::vector<ShapeVector> &shapes) {
  std::string shape_key = GenerateShapeKey(shapes);
  return Resize(shapes, shape_key);
}

Status Dequantizer::Resize(std::vector<ShapeVector> &shapes, const std::string &shape_key) {
  if (auto it = graph_cache_.find(shape_key); it != graph_cache_.end()) {
    MS_LOG(WARNING) << "Dequantizer: KernelGraph for shape " << shape_key << " has been generated";
    return SUCCESS; 
  }
  const auto &inputs = model_->parameters();
  if (inputs.size() != shapes.size()) {
    MS_LOG(EXCEPTION) << "Dequantizer: shapes size " << shapes.size() << " does not match with parameter numbers " << inputs.size();
    return FAILED;
  }
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto &param = inputs[i];
    auto shape_ptr = std::dynamic_pointer_cast<abstract::Shape>(param->Shape());
    if (shape_ptr == nullptr) {
      MS_LOG(ERROR) << "Dequantizer: inputs " << i << " is not supported to resize, debug string: " << param->DebugString();
      return FAILED;
    }
    shape_ptr->set_shape(shapes[i]);
  }
  auto graph = Compile();
  if (graph == nullptr) {
    MS_LOG(EXCEPTION) << "Dequantizer: Compile failed for shape " << shape_key;
    return FAILED;
  }
  graph_cache_[shape_key] = graph;
  return SUCCESS;
}

tensor::TensorPtrList Dequantizer::Dequantize(KernelGraphPtr graph, const tensor::TensorPtrList &inputs) {
  if (inputs.size() != 3) {
    MS_LOG(EXCEPTION) << "Dequantizer: input number should be exactly 3";
    return {};
  }
  VectorRef outputs;
  session_impl_->RunGraphAsync(graph->graph_id(), inputs, &outputs);
  auto ret = TransformVectorRefToMultiTensor(outputs);
  return ret;
}

KernelGraphPtr Dequantizer::GetKernelGraphByShape(std::vector<ShapeVector> &shapes) {
  std::string shape_key = GenerateShapeKey(shapes);
  auto it = graph_cache_.find(shape_key);
  if (it == graph_cache_.end()) {
    Resize(shapes);
    return graph_cache_.find(shape_key)->second; 
  }
  return it->second;
}

void TensorToProtoTensorCompressed(Compressor& compressor, const tensor::TensorPtr &tensor, offloading_serving::TensorProto *const tensor_proto) {
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
  auto csize = compressor.Compress(tensor->data_c(), abstract::TypeIdSize(tensor->data_type()), tensor->data().nbytes());
  tensor_proto->set_raw_data(compressor.GetBuffer(), csize);
  tensor_proto->set_compressed(true);
}

tensor::TensorPtr ProtoTensorToTensorDecompressed(Decompressor& decompressor, const offloading_serving::TensorProto &tensor_proto) {
  ShapeVector shape;
  for (int i = 0; i < tensor_proto.dims_size(); ++i) {
    shape.push_back(tensor_proto.dims(i));
  }

  auto dtype = GetTypeIdFromProtoTensor(tensor_proto);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(dtype, shape);
  const std::string &init_data = tensor_proto.raw_data();
  if (tensor_proto.compressed()) {
    auto *init_data_c = reinterpret_cast<const void *>(init_data.data());
    MS_EXCEPTION_IF_NULL(init_data_c);
    auto *data_buf = reinterpret_cast<void *>(tensor->data_c());
    MS_EXCEPTION_IF_NULL(data_buf);
    auto ret = decompressor.Decompress(init_data_c, init_data.size(), data_buf, tensor->data().nbytes());
    if (ret < 0) {
      return nullptr;
    }
  } else {
    MS_LOG(WARNING) << "Calling ProtoTensorToTensorDecompressed while the proto tensor is not compressed";
    auto *data_buf = reinterpret_cast<uint8_t *>(tensor->data_c());
    MS_EXCEPTION_IF_NULL(data_buf);
    auto ret = memcpy_s(data_buf, tensor->data().nbytes(), init_data.data(), init_data.size());
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error for building Tensor from TensorProto, errorno " << ret;
      return nullptr;
    }
  }
  return tensor;
}

}
}
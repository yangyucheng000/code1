#ifndef MINDSPORE_OFFLOADING_QUANT_H
#define MINDSPORE_OFFLOADING_QUANT_H

#include "blosc2.h"
#include "utils.h"
#include "status.h"

namespace mindspore {

namespace offloading {

constexpr size_t COMPRESS_BUFFER_SIZE = (1 << 20);

// Compressor is not thread-safe!
class Compressor {
  public:
    Compressor(int clevel, int do_shuffle, int nthreads, std::string compressor);
    ~Compressor();
    size_t Compress(const void *src, size_t typesize, size_t nbytes);
    void* GetBuffer();
  private:
    int clevel_;
    int do_shuffle_;
    int nthreads_;
    std::string compressor_;
    size_t buffer_size_;
    void* buffer_;
    // for logging
    std::ofstream log_file;
};

class Decompressor {
  public:
    Decompressor(int nthreads);
    ~Decompressor();
    size_t Decompress(const void *src, size_t nbytes, void *dest, size_t destsize);
  private:
    int nthreads_;
};

class Quantizer {
  public:
    Quantizer(const std::string &path, session::SessionPtr session_impl);
    ~Quantizer() = default;
    Status Load(const std::string &path);
    KernelGraphPtr Compile(size_t dim);
    Status Resize(std::vector<ShapeVector> &shapes);
    Status Resize(std::vector<ShapeVector> &shapes, const std::string &shapes_key);
    tensor::TensorPtrList Quantize(KernelGraphPtr graph, const tensor::TensorPtrList &inputs);
    KernelGraphPtr GetKernelGraphByShape(std::vector<ShapeVector> &shapes);
  private:
    std::unordered_map<std::string, KernelGraphPtr> graph_cache_;
    session::SessionPtr session_impl_ = nullptr;
    std::unordered_map<size_t, FuncGraphPtr> models_;
    std::unordered_map<size_t, FuncGraphManagerPtr> model_managers_;
};

class Dequantizer {
  public:
    Dequantizer(const std::string &path, session::SessionPtr session_impl);
    ~Dequantizer() = default;
    Status Load(const std::string &path);
    KernelGraphPtr Compile();
    Status Resize(std::vector<ShapeVector> &shapes);
    Status Resize(std::vector<ShapeVector> &shapes, const std::string &shapes_key);
    tensor::TensorPtrList Dequantize(KernelGraphPtr graph, const tensor::TensorPtrList &inputs);
    KernelGraphPtr GetKernelGraphByShape(std::vector<ShapeVector> &shapes);
  private:
    std::unordered_map<std::string, KernelGraphPtr> graph_cache_;
    session::SessionPtr session_impl_ = nullptr;
    FuncGraphPtr model_ = nullptr;
    FuncGraphManagerPtr model_manager_ = nullptr;
};

tensor::TensorPtr ProtoTensorToTensorDecompressed(Decompressor& decompressor, const offloading_serving::TensorProto &tensor_proto);

void TensorToProtoTensorCompressed(Compressor& compressor, const tensor::TensorPtr &tensor, offloading_serving::TensorProto *const tensor_proto);

}
}

#endif
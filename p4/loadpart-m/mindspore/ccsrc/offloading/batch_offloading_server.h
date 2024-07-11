#ifndef MINDSPORE_BATCH_OFFLOADING_SERVER_H
#define MINDSPORE_BATCH_OFFLOADING_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include "status.h"
#include "utils.h"
#include "cost_graph.h"
#include "quant.h"
#include "grpc_async_server.h"
#include "scheduler.h"

using offloading_serving::BatchOffloadingService;
using offloading_serving::PredictRequest;
using offloading_serving::PredictReply;
using offloading_serving::DeployRequest;
using offloading_serving::DeployReply;
using offloading_serving::ProfileRequest;
using offloading_serving::ProfileReply;
using offloading_serving::GraphProfile;

namespace mindspore {

namespace offloading {

namespace py = pybind11;

class BatchOffloadingServiceImpl {
  public:
    explicit BatchOffloadingServiceImpl(OffloadingServerConfig config) : cfg_(config), sched_(std::make_shared<Scheduler>(static_cast<ExecutionMode>(config.exec_mode), config.max_wait_time_ms, config.max_batch_size, config.load_window_size)) {}
    ~BatchOffloadingServiceImpl() = default;
    void PredictAsync(const offloading_serving::PredictRequest *request, offloading_serving::PredictReply *reply, PredictOnFinish on_finish);
    grpc::Status Deploy(const offloading_serving::DeployRequest *request, offloading_serving::DeployReply *reply);
    grpc::Status Profile(const offloading_serving::ProfileRequest *request, offloading_serving::ProfileReply *reply);
    grpc::Status TestUploadBandwidth(const offloading_serving::SimpleRequest *request, offloading_serving::SimpleReply *reply);
    grpc::Status TestDownloadBandwidth(const offloading_serving::SimpleRequest *request, offloading_serving::SimpleReply *reply);
    // void Clear();
  private:
    bool ProfileGraph(GraphId graph_id, KernelGraphPtr &graph, session::SessionPtr &session_impl, std::unordered_map<std::string, float> &result);
    
    OffloadingServerConfig cfg_;
    std::shared_ptr<Scheduler> sched_;
    bool is_init_ = false;
    bool is_deployed_ = false;
};

template <class Derived>
class BatchOffloadingGrpcContext : public GrpcAsyncServiceContext<BatchOffloadingServiceImpl, offloading_serving::BatchOffloadingService::AsyncService, Derived> {
  public:
    BatchOffloadingGrpcContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                          grpc::ServerCompletionQueue *cq)
      : GrpcAsyncServiceContext<BatchOffloadingServiceImpl, offloading_serving::BatchOffloadingService::AsyncService, Derived>(service_impl, async_service, cq) {}
    
    virtual void StartEnqueueRequest() = 0;
    virtual void HandleRequest() = 0;
};

class BatchOffloadingPredictContext : public BatchOffloadingGrpcContext<BatchOffloadingPredictContext> {
  public:
    BatchOffloadingPredictContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : BatchOffloadingGrpcContext<BatchOffloadingPredictContext>(service_impl, async_service, cq), responder_(&ctx_) {}
    
    ~BatchOffloadingPredictContext() = default;
    
    void StartEnqueueRequest() override { 
      async_service_->RequestPredict(&ctx_, &request_, &responder_, cq_, cq_, this); 
    }
    
    void HandleRequest() override {
      PredictOnFinish on_finish = [this](offloading_serving::PredictReply* reply, grpc::Status status){ responder_.Finish(*reply, status, this); };
      service_impl_->PredictAsync(&request_, &response_, on_finish);
    }
  
  private:
    grpc::ServerAsyncResponseWriter<offloading_serving::PredictReply> responder_;
    offloading_serving::PredictRequest request_;
    offloading_serving::PredictReply response_;
};

class BatchOffloadingDeployContext : public BatchOffloadingGrpcContext<BatchOffloadingDeployContext> {
  public:
    BatchOffloadingDeployContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : BatchOffloadingGrpcContext<BatchOffloadingDeployContext>(service_impl, async_service, cq), responder_(&ctx_) {}
    
    ~BatchOffloadingDeployContext() = default;
    
    void StartEnqueueRequest() override { 
      async_service_->RequestDeploy(&ctx_, &request_, &responder_, cq_, cq_, this); 
    }
    
    void HandleRequest() override {
      grpc::Status status = service_impl_->Deploy(&request_, &response_);
      responder_.Finish(response_, status, this);
    }
  
  private:
    grpc::ServerAsyncResponseWriter<offloading_serving::DeployReply> responder_;
    offloading_serving::DeployRequest request_;
    offloading_serving::DeployReply response_;
};

class BatchOffloadingProfileContext : public BatchOffloadingGrpcContext<BatchOffloadingProfileContext> {
  public:
    BatchOffloadingProfileContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : BatchOffloadingGrpcContext<BatchOffloadingProfileContext>(service_impl, async_service, cq), responder_(&ctx_) {}
    
    ~BatchOffloadingProfileContext() = default;
    
    void StartEnqueueRequest() override { 
      async_service_->RequestProfile(&ctx_, &request_, &responder_, cq_, cq_, this); 
    }
    
    void HandleRequest() override {
      grpc::Status status = service_impl_->Profile(&request_, &response_);
      responder_.Finish(response_, status, this);
    }
  
  private:
    grpc::ServerAsyncResponseWriter<offloading_serving::ProfileReply> responder_;
    offloading_serving::ProfileRequest request_;
    offloading_serving::ProfileReply response_;
};

class BatchOffloadingTestUploadBandwidthContext : public BatchOffloadingGrpcContext<BatchOffloadingTestUploadBandwidthContext> {
  public:
    BatchOffloadingTestUploadBandwidthContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : BatchOffloadingGrpcContext<BatchOffloadingTestUploadBandwidthContext>(service_impl, async_service, cq), responder_(&ctx_) {}
    
    ~BatchOffloadingTestUploadBandwidthContext() = default;
    
    void StartEnqueueRequest() override { 
      async_service_->RequestTestUploadBandwidth(&ctx_, &request_, &responder_, cq_, cq_, this); 
    }
    
    void HandleRequest() override {
      grpc::Status status = service_impl_->TestUploadBandwidth(&request_, &response_);
      responder_.Finish(response_, status, this);
    }
  
  private:
    grpc::ServerAsyncResponseWriter<offloading_serving::SimpleReply> responder_;
    offloading_serving::SimpleRequest request_;
    offloading_serving::SimpleReply response_;
};

class BatchOffloadingTestDownloadBandwidthContext : public BatchOffloadingGrpcContext<BatchOffloadingTestDownloadBandwidthContext> {
  public:
    BatchOffloadingTestDownloadBandwidthContext(BatchOffloadingServiceImpl *service_impl, offloading_serving::BatchOffloadingService::AsyncService *async_service,
                             grpc::ServerCompletionQueue *cq)
      : BatchOffloadingGrpcContext<BatchOffloadingTestDownloadBandwidthContext>(service_impl, async_service, cq), responder_(&ctx_) {}
    
    ~BatchOffloadingTestDownloadBandwidthContext() = default;
    
    void StartEnqueueRequest() override { 
      async_service_->RequestTestDownloadBandwidth(&ctx_, &request_, &responder_, cq_, cq_, this); 
    }
    
    void HandleRequest() override {
      grpc::Status status = service_impl_->TestDownloadBandwidth(&request_, &response_);
      responder_.Finish(response_, status, this);
    }
  
  private:
    grpc::ServerAsyncResponseWriter<offloading_serving::SimpleReply> responder_;
    offloading_serving::SimpleRequest request_;
    offloading_serving::SimpleReply response_;
};

class BatchOffloadingGrpcServer : public GrpcAsyncServer<offloading_serving::BatchOffloadingService::AsyncService> {
  public:
    explicit BatchOffloadingGrpcServer(OffloadingServerConfig config)
      : GrpcAsyncServer<offloading_serving::BatchOffloadingService::AsyncService>(), service_impl_(config) {}
    ~BatchOffloadingGrpcServer() {}
    void EnqueueRequests() override {
      BatchOffloadingPredictContext::EnqueueRequest(&service_impl_, &svc_, offloading_cq_.get());
      BatchOffloadingDeployContext::EnqueueRequest(&service_impl_, &svc_, offloading_cq_.get());
      BatchOffloadingProfileContext::EnqueueRequest(&service_impl_, &svc_, offloading_cq_.get());
      BatchOffloadingTestUploadBandwidthContext::EnqueueRequest(&service_impl_, &svc_, profile_cq_.get());
      BatchOffloadingTestDownloadBandwidthContext::EnqueueRequest(&service_impl_, &svc_, profile_cq_.get());
    }
  protected:
    BatchOffloadingServiceImpl service_impl_;
};

class BatchOffloadingServer {
  public:
    static BatchOffloadingServer &GetInstance();
    BatchOffloadingServer() = default;
    ~BatchOffloadingServer() = default;
    Status StartGrpcServer(const std::string &ip, uint32_t grpc_port, OffloadingServerConfig config, int max_msg_mb_size);
    void Stop();
  private:
    std::shared_ptr<BatchOffloadingGrpcServer> grpc_async_server_;
};

}
}

#endif
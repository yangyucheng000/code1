#ifndef MINDSPORE_OFFLOADING_GRPC_ASYNC_SERVER_H
#define MINDSPORE_OFFLOADING_GRPC_ASYNC_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <memory>
#include <utility>
#include <string>
#include <future>
#include "utils/log_adapter.h"
#include "status.h"

namespace mindspore::offloading {

constexpr int gRpcDefaultMsgMBSize = 100;
constexpr int gRpcMaxMBMsgSize = 512;  // max 512 MB

class GrpcAsyncServiceContextBase {
 public:
  GrpcAsyncServiceContextBase() = default;
  virtual ~GrpcAsyncServiceContextBase() = default;

  virtual void NewAndHandleRequest() = 0;

  bool HasFinish() const { return finished_; }
  void SetFinish() { finished_ = true; }

 private:
  bool finished_ = false;
};

template <class ServiceImpl, class AsyncService, class Derived>
class GrpcAsyncServiceContext : public GrpcAsyncServiceContextBase {
 public:
  GrpcAsyncServiceContext(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq)
      : service_impl_(service_impl), async_service_(async_service), cq_(cq) {}
  ~GrpcAsyncServiceContext() = default;
  GrpcAsyncServiceContext() = delete;

  virtual void StartEnqueueRequest() = 0;
  virtual void HandleRequest() = 0;

  static void EnqueueRequest(ServiceImpl *service_impl, AsyncService *async_service, grpc::ServerCompletionQueue *cq) {
    auto call = new Derived(service_impl, async_service, cq);
    call->StartEnqueueRequest();
  }

  void NewAndHandleRequest() final {
    EnqueueRequest(service_impl_, async_service_, cq_);
    HandleRequest();
  }

 protected:
  grpc::ServerContext ctx_;

  ServiceImpl *service_impl_;
  AsyncService *async_service_;
  grpc::ServerCompletionQueue *cq_;
};

template <class AsyncService>
class GrpcAsyncServer {
 public:
  GrpcAsyncServer() {}
  virtual ~GrpcAsyncServer() { Stop(); }

  virtual void EnqueueRequests() = 0;

  Status Start(const std::string &socket_address, int max_msg_mb_size,
               const std::string &server_tag) {
    if (offloading_in_running_ || profile_in_running_) {
      MS_LOG(ERROR) << "Serving Error: " << server_tag << " server is already running";
      return FAILED;
    }

    grpc::ServerBuilder builder;
    if (max_msg_mb_size > 0) {
      builder.SetMaxSendMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
      builder.SetMaxReceiveMessageSize(static_cast<int>(max_msg_mb_size * (1u << 20)));
    }
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    int port_tcpip = 0;
    auto creds = grpc::InsecureServerCredentials();

    Status status;
    builder.AddListeningPort(socket_address, creds, &port_tcpip);
    status = RegisterService(&builder);
    if (status != SUCCESS) return status;
    offloading_cq_ = builder.AddCompletionQueue();
    profile_cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    if (!server_) {
      MS_LOG(ERROR) << "Serving Error: " << server_tag
                                         << " server start failed, create server failed, address " << socket_address;
      return FAILED;
    }
    auto grpc_offloading_run = [this]() { HandleRequests(offloading_cq_.get()); };
    grpc_offloading_thread_ = std::thread(grpc_offloading_run);
    offloading_in_running_ = true;
    auto grpc_profile_run = [this]() { HandleRequests(profile_cq_.get()); };
    grpc_profile_thread_ = std::thread(grpc_profile_run);
    profile_in_running_ = true;
    MS_LOG(INFO) << server_tag << " server start success, listening on " << socket_address;
    std::cout << "Serving: " << server_tag << " server start success, listening on " << socket_address << std::endl;
    return SUCCESS;
  }

  Status HandleRequests(grpc::CompletionQueue* cq) {
    void *tag;
    bool ok = false;
    EnqueueRequests();
    while (cq->Next(&tag, &ok)) {
      ProcessRequest(tag, ok);
    }
    return SUCCESS;
  }

  void Stop() {
    if (offloading_in_running_ || profile_in_running_) {
      if (server_) {
        server_->Shutdown();
      }
      // Always shutdown the completion queue after the server.
      if (offloading_cq_) {
        offloading_cq_->Shutdown();
      }
      if (profile_cq_) {
        profile_cq_->Shutdown();
      }
      grpc_offloading_thread_.join();
      grpc_profile_thread_.join();
    }
    offloading_in_running_ = false;
    profile_in_running_ = false;
  }

  Status RegisterService(grpc::ServerBuilder *builder) {
    builder->RegisterService(&svc_);
    return SUCCESS;
  }

  void ProcessRequest(void *tag, bool rpc_ok) {
    auto rq = static_cast<GrpcAsyncServiceContextBase *>(tag);
    if (rq->HasFinish() || !rpc_ok) {  // !rpc_ok: cancel get request when shutting down.
      delete rq;
    } else {
      rq->NewAndHandleRequest();
      rq->SetFinish();  // will delete next time
    }
  }

 protected:
  std::unique_ptr<grpc::ServerCompletionQueue> offloading_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> profile_cq_;
  std::unique_ptr<grpc::Server> server_;

  AsyncService svc_;

  bool offloading_in_running_ = false;
  bool profile_in_running_ = false;
  std::thread grpc_offloading_thread_;
  std::thread grpc_profile_thread_;
};

}  // namespace mindspore::offloading

#endif  // MINDSPORE_OFFLOADING_GRPC_ASYNC_SERVER_H
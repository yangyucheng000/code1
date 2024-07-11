#include "offloading_client_py.h"

namespace mindspore {
  
namespace offloading {

std::unique_ptr<OffloadingClient> OffloadingClientPy::NewClient(const std::string &addr, OffloadingClientConfig config) {
  grpc::ChannelArguments ch_args;
  int max_msg_mb_size = config.max_msg_mb_size < 0 ? -1 : config.max_msg_mb_size * (1 << 20);
  ch_args.SetMaxReceiveMessageSize(max_msg_mb_size);
  std::unique_ptr<OffloadingClient> c(new OffloadingClient(grpc::CreateCustomChannel(addr, grpc::InsecureChannelCredentials(), ch_args), config));
  return c;
}

void OffloadingClientPy::StartGrpcClient(const std::string &ip, uint32_t grpc_port, OffloadingClientConfig config) {
  std::string server_address = ip + ":" + std::to_string(grpc_port);
  client_ = NewClient(server_address, config);
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client start failed";
  }
  MS_LOG(INFO) << "gRPC client start successfully, connecting to " << server_address;
  std::cout << "gRPC client start successfully, connecting to " << server_address << std::endl;
}

void OffloadingClientPy::Clear() {
  client_->Clear();
  return;
}

void OffloadingClientPy::Stop() {
  Clear();
  return;
}

py::object OffloadingClientPy::Predict(const std::unordered_map<std::string, py::object> &kwargs) {
  py::object ret;
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return ret;
  }
  ret = client_->Predict(kwargs);
  return ret;
}

py::object OffloadingClientPy::PredictOracle(const std::unordered_map<std::string, py::object> &kwargs, double bdw, double q_time, double load_factor) {
  py::object ret;
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return ret;
  }
  ret = client_->PredictOracle(kwargs, bdw, q_time, load_factor);
  return ret;
}

py::object OffloadingClientPy::StaticPredictTest(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  py::object ret;
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return ret;
  }
  ret = client_->StaticPredictTest(lg_idx, cut_nodes_names, kwargs);
  return ret;
}

int OffloadingClientPy::PrintCNodeOrder() {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return -1;
  }
  return client_->PrintCNodeOrder();
}

void OffloadingClientPy::Deploy(const std::string &path) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return;
  }
  client_->Deploy(path);
}

void OffloadingClientPy::Profile(const std::string &path) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return;
  }
  client_->Profile(path);
}

void OffloadingClientPy::PartitionDecisionTest(double bdw, double q_time, double load_factor) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return;
  }
  client_->PartitionDecisionTest(bdw, q_time, load_factor);
}

void OffloadingClientPy::FakeDeploy(const std::string &path) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return;
  }
  client_->FakeDeploy(path);
}

void OffloadingClientPy::FakeStaticPredictTest(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return;
  }
  client_->FakeStaticPredictTest(lg_idx, cut_nodes_names, kwargs);
}

py::object OffloadingClientPy::PartialExecute(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return py::none();
  }
  return client_->PartialExecute(lg_idx, cut_nodes_names, kwargs);
}

py::object OffloadingClientPy::ConvertProtoTensorFromFile(const std::string &path) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return py::none();
  }
  return client_->ConvertProtoTensorFromFile(path);
}

py::object OffloadingClientPy::UnitTest(const std::unordered_map<std::string, py::object> &kwargs) {
  if (client_ == nullptr) {
    MS_LOG(EXCEPTION) << "gRPC client not started, please call start_grpc_client";
    return py::none();
  }
  return client_->UnitTest(kwargs);
}

}
}
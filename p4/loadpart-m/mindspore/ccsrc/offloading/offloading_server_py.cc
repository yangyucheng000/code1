#include "offloading_server_py.h"

namespace mindspore {

namespace offloading {

void BatchOffloadingServerPy::StartGrpcServer(const std::string &ip, uint32_t grpc_port, OffloadingServerConfig config) {
  auto status = BatchOffloadingServer::GetInstance().StartGrpcServer(ip, grpc_port, config, config.max_msg_mb_size);
  if (status != SUCCESS) {
    MS_LOG(EXCEPTION) << "Raise failed";
  }
}

void BatchOffloadingServerPy::Stop() {
  BatchOffloadingServer::GetInstance().Stop();
}

}
    
} // namespace mindspore

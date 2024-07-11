#ifndef MINDSPORE_OFFLOADING_SERVER_PY_H
#define MINDSPORE_OFFLOADING_SERVER_PY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include "batch_offloading_server.h"
#include "utils.h"

namespace mindspore {

namespace offloading {

namespace py = pybind11;

class BatchOffloadingServerPy {
  public:
    void StartGrpcServer(const std::string &ip, uint32_t grpc_port, OffloadingServerConfig config);
    void Stop();
};

}

}

#endif
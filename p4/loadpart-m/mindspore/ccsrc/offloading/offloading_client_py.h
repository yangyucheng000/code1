#ifndef MINDSPORE_OFFLOADING_CLIENT_PY_H
#define MINDSPORE_OFFLOADING_CLIENT_PY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include "offloading_client.h"
#include "utils.h"

namespace mindspore {

namespace offloading {

namespace py = pybind11;

class OffloadingClientPy {
  public:
    void StartGrpcClient(const std::string &ip, uint32_t grpc_port, OffloadingClientConfig config);
    void Clear();
    void Stop();
    py::object Predict(const std::unordered_map<std::string, py::object> &kwargs);
    py::object PredictOracle(const std::unordered_map<std::string, py::object> &kwargs, double bdw, double q_time, double load_factor);
    py::object StaticPredictTest(const size_t lg_idx, const std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    int PrintCNodeOrder();
    void Deploy(const std::string &path);
    void Profile(const std::string &path);
    // for experiment
    void PartitionDecisionTest(double bdw, double q_time, double load_factor);
    void FakeDeploy(const std::string &path);
    void FakeStaticPredictTest(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    py::object PartialExecute(const size_t lg_idx, std::vector<std::string> &cut_nodes_names, const std::unordered_map<std::string, py::object> &kwargs);
    py::object ConvertProtoTensorFromFile(const std::string &path);
    // for test
    py::object UnitTest(const std::unordered_map<std::string, py::object> &kwargs);
  private:
    std::unique_ptr<OffloadingClient> client_;
    static std::unique_ptr<OffloadingClient> NewClient(const std::string &addr, OffloadingClientConfig config);
};

}

}

#endif
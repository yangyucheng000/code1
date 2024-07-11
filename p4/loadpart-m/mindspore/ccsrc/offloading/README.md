# Load-aware Dynamic DNN Partition for Multi-User Offloading

We design and implement a load-aware deep neural network partition and offloading system for multi-user offloading. In the system, the computation resources of a server are shared by multiple user-end devices for inference. These user-end devices can dynamically and jointly analyze both the available network bandwidth and the load of the server and make proper decisions on deep neural network partition to minimize the end-to-end inference latency.

This directory is the codebase of the system. The contents of the files are:

- `batch_offloading_server.h` & `.cc`: `BatchOffloadingServer` and `BatchOffloadingServiceImpl` for serving offloading requests asynchronously.
- `cost_graph.h` & `.cc`: platform-agnostic `CostGraph` and `LatencyGraph` for partition decision algorithms.
- `grpc_async_server.h`: the definition of an asynchronous gRPC server.
- `offloading_client.h` & `.cc`: `OffloadingClient` for inference execution and offloading.
- `offloading_client_py.h` & `.cc`: wrapped `OffloadingClient` for `pybind11` binding.
- `offloading_server_py.h` & `.cc`: wrapped `BatchOffloadingServer` for `pybind11` binding.
- `offloading_service.proto`: offloading service-related protobuf definition.
- `payload.h` & `.cc`: `Payload` used by `Scheduler` for packing incoming request data.
- `quant.h` & `.cc`: `Quantizer` and `Dequantizer` for intermediate results (de-)quantization together with `Compressor` and `Decompressor` for the (de-)quantized results.
- `scheduler.h` & `.cc`: `Scheduler` for FCFS scheduling and batching on the server side.
- `stage_executor.h` & `.cc`: `StageExecutor`s for request data preprocessing and inference execution.
- `status.h`: the definition of status used in gRPC-related classes/functions.
- `task_queue.h` & `.cc`: `Task` and `TaskQueue` for passing data between `StageExecutor`s.
- `utils.h` & `.cc`: the definition and implementation of auxiliary classes and functions.

Please follow [this documentation](https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_gpu_install_source.md) for building MindSpore from source.
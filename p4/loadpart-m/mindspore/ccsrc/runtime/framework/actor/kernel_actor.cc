/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/recorder_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void KernelActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  MS_EXCEPTION_IF_NULL(kernel_);
  real_input_num_ = AnfAlgo::GetInputTensorNum(kernel_);
  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  is_dynamic_shape_ = AnfAlgo::IsDynamicShape(kernel_);

  // Init the device tensors and kernel launch info.
  copy_input_device_tensors_.resize(real_input_num_);
  input_device_tensors_.resize(real_input_num_);
  for (auto &input_address : input_device_tensors_) {
    (void)memory_free_list_.emplace_back(input_address);
    (void)launch_info_.inputs_.emplace_back(std::make_shared<Address>());
  }
  MS_EXCEPTION_IF_NULL(kernel_info_);
  for (auto &output_address : kernel_info_->output_address_list()) {
    MS_EXCEPTION_IF_NULL(output_address);
    (void)output_device_tensors_.emplace_back(output_address.get());
    (void)memory_alloc_list_.emplace_back(output_address.get());
    (void)memory_free_list_.emplace_back(output_address.get());
    (void)launch_info_.outputs_.emplace_back(std::make_shared<Address>());
  }
  for (auto &workspace_address : kernel_info_->workspace_address_list()) {
    MS_EXCEPTION_IF_NULL(workspace_address);
    (void)workspace_device_tensors_.emplace_back(workspace_address.get());
    (void)memory_alloc_list_.emplace_back(workspace_address.get());
    (void)memory_free_list_.emplace_back(workspace_address.get());
    (void)launch_info_.workspaces_.emplace_back(std::make_shared<Address>());
  }
  for (auto &external_reference_tensor : external_reference_tensors_) {
    (void)memory_free_list_.emplace_back(external_reference_tensor);
  }

  // Init the output data.
  output_data_by_output_index_.resize(output_device_tensors_.size());
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    auto device_address = output_device_tensors_[data_arrow->from_output_index_];
    auto data =
      std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, device_address, data_arrow->to_input_index_);
    (void)output_data_by_output_index_[data_arrow->from_output_index_].emplace_back(data.get());
    (void)output_data_.emplace_back(std::move(data));
  }
}

void KernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  // Infer kernel shape and update abstract info for dynamic shape kernel.
  if (is_dynamic_shape_) {
    try {
      device_contexts_[0]->UpdateDynamicShape(kernel_);
    } catch (const std::exception &e) {
      if (strategy_ == GraphExecutionStrategy::kPipeline) {
        MsException::Instance().SetException();
      }
      std::string error_info = "Update Dynamic shape exception: " + kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }
  }

  FetchInputDeviceTensor(context);
  FetchOutputDeviceTensor(context);
  if (memory_alloc_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
}

void KernelActor::RunOpControlWithInputTensor(AID *const input_control, OpContext<DeviceTensor> *const context,
                                              const std::vector<TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_tensors);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);

  PushInputDeviceTensor(input_tensors);
  // When all the inputs are collected, then allocate memory and callback launch.
  if (CheckRunningCondition(context)) {
    if (is_dynamic_shape_) {
      device_contexts_[0]->UpdateDynamicShape(kernel_);
    }

    FetchOutputDeviceTensor(context);
    if (memory_alloc_list_.size() > 0) {
      SendMemoryAllocReq(context);
    }
    OnMemoryAllocFinish(context);
  }
}

namespace {
void AllocateMemory(const std::vector<DeviceTensor *> &alloc_list, const DeviceContext *device_context,
                    OpContext<DeviceTensor> *const context, const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);

  for (auto &device_tensor : alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if ((device_tensor->GetPtr() != nullptr) || (device_tensor->GetSize() == 0)) {
      continue;
    }
    // Allocate memory through the device context.
    if (!device_context->AllocateMemory(device_tensor, device_tensor->GetSize())) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kStep, *context, *device_context, actor_name,
                                                  device_tensor->GetSize());
    }
  }
}

void FreeMemory(const std::vector<DeviceTensor *> &free_list, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  for (auto &device_tensor : free_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->original_ref_count() == SIZE_MAX) {
      continue;
    }
    // The reference count is decremented to zero to free memory, and reset to the original count.
    device_tensor->DecreaseRefCount();
    if (device_tensor->ref_count() == 0) {
      // Free memory through the device context.
      if (device_tensor->GetPtr() != nullptr) {
        device_context->FreeMemory(device_tensor);
      }
      device_tensor->ResetRefCount();
    }
  }
}
}  // namespace

void KernelActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                          device_contexts_[0], context, GetAID());
  } else {
    AllocateMemory(memory_alloc_list_, device_contexts_[0], context, GetAID().Name());
  }
}

void KernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list_, device_contexts_[0],
                          context, GetAID());
  } else {
    FreeMemory(memory_free_list_, device_contexts_[0]);
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_device_tensor : copy_input_device_tensors_) {
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->FreeMemory(copy_input_device_tensor.get());
    }
  }
}

void KernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  PreLaunchKernel(context);

  try {
    auto ret = device_contexts_[0]->LaunchKernel(kernel_, launch_info_.inputs_, launch_info_.workspaces_,
                                                 launch_info_.outputs_, is_dynamic_shape_);
    if (!ret) {
      std::string error_info = "Launch kernel failed: " + kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }
  } catch (const std::exception &e) {
    if (strategy_ == GraphExecutionStrategy::kPipeline) {
      MsException::Instance().SetException();
    }
    std::string error_info = "Launch kernel exception: " + kernel_->fullname_with_scope();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendDebugReq(context);
    return;
  }

  PostLaunchKernel(context);
}

void KernelActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  ActorDispatcher::Send(*debug_aid_, &DebugActor::Debug, kernel_, &launch_info_, device_contexts_[0], context,
                        &GetAID());
}

void KernelActor::OnDebugFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  PostLaunchKernel(context);
}

void KernelActor::PushInputDeviceTensor(const std::vector<TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (input_tensors->size() != real_input_num_) {
    MS_LOG(ERROR) << "Input tensor number: " << input_tensors->size()
                  << " is not equal to kernel's input size: " << real_input_num_;
    return;
  }

  for (size_t input_index = 0; input_index < input_tensors->size(); input_index++) {
    const auto &input_tensor = (*input_tensors)[input_index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    const auto &device_tensor = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
    if (device_tensor != nullptr) {
      input_device_tensors_[input_index] = device_tensor.get();
      memory_free_list_[input_index] = device_tensor.get();
    }
  }
}

void KernelActor::CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data,
                                        OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  const auto &device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_, input_data->index_, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if ((input_data->data_->DeviceType() == device_tensor->DeviceType()) &&
      (input_data->data_->format() == device_tensor->format())) {
    return;
  }

  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  if (IntToSize(input_data->index_) >= copy_input_device_tensors_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
  }
  if (copy_input_device_tensors_[input_data->index_] == nullptr) {
    copy_input_device_tensors_[input_data->index_] = device_contexts_[0]->CreateDeviceAddress(
      nullptr, device_tensor->GetSize(), device_tensor->format(), device_tensor->type_id());
  }
  auto &new_device_tensor = copy_input_device_tensors_[input_data->index_];
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  // Dynamic shape need update size.
  new_device_tensor->SetSize(input_data->data_->GetSize());
  // Update the input device tensor.
  input_device_tensors_[input_data->index_] = new_device_tensor.get();

  if ((new_device_tensor->GetPtr() == nullptr) &&
      (!device_contexts_[0]->AllocateMemory(new_device_tensor.get(), new_device_tensor->GetSize()))) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *(device_contexts_[0]), GetAID().Name(),
                                                new_device_tensor->GetSize());
  }
  MS_LOG(INFO) << GetAID().Name() << " the input position:" << input_data->index_
               << " copy from device address:" << input_data->data_ << ", type:" << input_data->data_->DeviceType()
               << ", format:" << input_data->data_->format() << " to device address:" << new_device_tensor.get()
               << ", type:" << new_device_tensor->DeviceType() << ", format:" << new_device_tensor->format();
  // Copy from the real parameter to formal parameter and insert the device tensor copy store.
  if (!Copy(new_device_tensor.get(), input_data->data_)) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (modifiable_ref_input_indexes_.count(input_data->index_) > 0) {
    DeviceTensorCopyStore::GetInstance().Insert(new_device_tensor.get(), input_data->data_);
  }
}

void KernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), "The input index is out of range.");
      }

      if (input_device_tensors_[input_data->index_] != input_data->data_) {
        input_device_tensors_[input_data->index_] = input_data->data_;
        memory_free_list_[input_data->index_] = input_data->data_;
      }
      CopyInputDeviceTensor(input_data, context);
    }
  }

  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceAddressType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }

    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), "The input index is out of range.");
    }
    if (input_device_tensors_[device_tensor_store_key.first] != device_tensor) {
      input_device_tensors_[device_tensor_store_key.first] = device_tensor;
      memory_free_list_[device_tensor_store_key.first] = device_tensor;
    }
  }
}

void KernelActor::FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  auto &output_addresses = kernel_info_->output_address_list();
  const auto &kernel_mod = kernel_info_->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto &output_size_list = kernel_mod->GetOutputSizeList();

  // May exist in the kernel which does not support the dynamic shape.
  if (output_addresses.size() != output_size_list.size()) {
    std::string error_info = "The outputs number(" + std::to_string(output_size_list.size()) + ") is wrong, " +
                             GetAID().Name() + " may not support the dynamic shape, please check.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto output_address = output_addresses[i].get();
    MS_EXCEPTION_IF_NULL(output_address);
    if (output_size_list[i] != output_address->GetSize()) {
      // The size of output address may be changed in dynamic shape scenario.
      // If the format of the DeviceAddress is different, then the size is originally different.
      // Such as NCHW(1,1,1,3) and NC1HWC0(1,1,1,1,16). So we don't need to update the size.
      if (AnfAlgo::GetOutputFormat(kernel_, i) == output_address->format()) {
        output_address->SetSize(output_size_list[i]);
      }
    }

    // When the tensor is the output of graph or in dynamic shape scenario, the output tensor may be changed.
    if (output_device_tensors_[i] != output_address) {
      output_device_tensors_[i] = output_address;
      memory_alloc_list_[i] = output_address;
      memory_free_list_[real_input_num_ + i] = output_address;

      // Update output data.
      for (auto &output_data : output_data_by_output_index_[i]) {
        MS_EXCEPTION_IF_NULL(output_data);
        output_data->data_ = output_address;
      }
    }
  }
}

void KernelActor::PreLaunchKernel(OpContext<DeviceTensor> *) {
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.inputs_[i]);
    launch_info_.inputs_[i]->addr = input_device_tensors_[i]->GetMutablePtr();
    launch_info_.inputs_[i]->size = input_device_tensors_[i]->GetSize();
  }

  for (size_t i = 0; i < output_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.outputs_[i]);
    launch_info_.outputs_[i]->addr = output_device_tensors_[i]->GetMutablePtr();
    launch_info_.outputs_[i]->size = output_device_tensors_[i]->GetSize();
  }

  for (size_t i = 0; i < workspace_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(workspace_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.workspaces_[i]);
    launch_info_.workspaces_[i]->addr = workspace_device_tensors_[i]->GetMutablePtr();
    launch_info_.workspaces_[i]->size = workspace_device_tensors_[i]->GetSize();
  }
}

void KernelActor::PostLaunchKernel(OpContext<DeviceTensor> *const context) {
  // The size of output address may be changed in dynamic shape scenario.
  if (is_dynamic_shape_) {
    UpdateOutputAddrSize();
  }

  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // The input is invalid and needs to be erased when finish kernel launch.
  EraseInput(context);

  // Note that SendMemoryFreeReq must be in front of SendOutput, because SendOutput will trigger SendMemoryAllocReq of
  // the next actor and the actor is asynchronous execution. So it is necessary to ensure that SendMemoryFreeReq of the
  // current actor is in front of SendMemoryAllocReq of the next actor.  One is to reuse the memory more fully, the
  // other is to ensure the execution order and avoid the illegal memory timing problem.
  if (memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }

  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    SendOutput(context);
  }
}

void KernelActor::UpdateOutputAddrSize() {
  auto &output_addresses = kernel_info_->output_address_list();
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto output_address = output_addresses[i].get();
    MS_EXCEPTION_IF_NULL(output_address);
    auto output_addr_size = AnfAlgo::GetOutputTensorMemSize(kernel_, i);
    if (output_addr_size != output_address->GetSize()) {
      output_address->SetSize(output_addr_size);
    }
  }
}

void KernelActor::RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  for (auto &ref_input_index : modifiable_ref_input_indexes_) {
    if (ref_input_index >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
    }
    auto &input_device_tensor = input_device_tensors_[ref_input_index];
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(input_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << ref_input_index
                   << " refresh from device address:" << input_device_tensor
                   << ", type:" << input_device_tensor->DeviceType() << ", format:" << input_device_tensor->format()
                   << " to device address:" << new_device_tensor << ", type:" << new_device_tensor->DeviceType()
                   << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, input_device_tensor)) {
        std::string error_info = "Copy input device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  for (auto &ref_output_index : modifiable_ref_output_indexes_) {
    if (ref_output_index >= output_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The output index is of range.");
    }
    auto &output_device_tensor = output_device_tensors_[ref_output_index];
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(output_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the output position:" << ref_output_index
                   << " refresh from device address:" << output_device_tensor
                   << ", type:" << output_device_tensor->DeviceType() << ", format:" << output_device_tensor->format()
                   << " to device address:" << new_device_tensor << ", type:" << new_device_tensor->DeviceType()
                   << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, output_device_tensor)) {
        std::string error_info = "Copy output device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }
}

void KernelActor::SendRecorderInfo(OpContext<DeviceTensor> *const context) const {
  if (recorder_aid_ != nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &launch_info_,
                          device_contexts_[0], context);
  }
}
}  // namespace runtime
}  // namespace mindspore

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
#include <cstdint>

#include "backend/kernel_compiler/gpu/arrays/dynamic_range_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      DynamicRangeGpuKernel, float)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      DynamicRangeGpuKernel, double)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      DynamicRangeGpuKernel, int32_t)

MS_REG_GPU_KERNEL_ONE(Range,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      DynamicRangeGpuKernel, int64_t)
}  // namespace kernel
}  // namespace mindspore

/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/fusion/l2_normalize_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void L2NormalizeFusion::Init(const std::vector<int64_t> &axis, const float epsilon,
                             const ActivationType &activation_type) {
  this->set_axis(axis);
  this->set_epsilon(epsilon);
  this->set_activation_type(activation_type);
}

void L2NormalizeFusion::set_activation_type(const ActivationType &activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, MakeValue(swi));
}

ActivationType L2NormalizeFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameL2NormalizeFusion, L2NormalizeFusion);
}  // namespace ops
}  // namespace mindspore
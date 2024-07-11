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

#include "ops/fusion/add_fusion.h"
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>

#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void AddFusion::set_activation_type(const ActivationType activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, MakeValue(swi));
}
ActivationType AddFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}
void AddFusion::Init(const ActivationType activation_type) { this->set_activation_type(activation_type); }

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("y", input_args[kInputIndex1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
}
}  // namespace

AbstractBasePtr AddFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameAddFusion, AddFusion);
}  // namespace ops
}  // namespace mindspore

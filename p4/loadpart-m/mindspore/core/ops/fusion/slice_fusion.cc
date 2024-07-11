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

#include "ops/fusion/slice_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void SliceFusion::Init(const std::vector<int64_t> &axes) { this->set_axes(axes); }

void SliceFusion::set_axes(const std::vector<int64_t> &axes) { (void)this->AddAttr(kAxes, MakeValue(axes)); }

std::vector<int64_t> SliceFusion::get_axes() const {
  auto value_ptr = GetAttr(kAxes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

AbstractBasePtr SliceFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape_len = x_shape.size();
  auto begin_v = input_args[kInputIndex1]->BuildValue();
  auto size_v = input_args[kInputIndex2]->BuildValue();
  auto x_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(begin_v);
  MS_EXCEPTION_IF_NULL(size_v);
  auto tensor_type = x_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  if (begin_v == kAnyValue || size_v == kAnyValue) {
    return std::make_shared<abstract::AbstractTensor>(data_type, std::vector<int64_t>{});
  }
  auto begin = GetValue<std::vector<int64_t>>(begin_v);
  auto size = GetValue<std::vector<int64_t>>(size_v);
  CheckAndConvertUtils::Check("len of begin", (int64_t)begin.size(), kEqual, SizeToLong(x_shape_len));
  CheckAndConvertUtils::Check("len of size", (int64_t)size.size(), kEqual, SizeToLong(x_shape_len));

  for (size_t i = 0; i < x_shape_len; i++) {
    (void)CheckAndConvertUtils::CheckInteger("input size[" + std::to_string(i) + "]", size[i], kGreaterThan, 0, "");
    if (x_shape[i] < (begin[i] + size[i])) {
      auto y = begin[i] + size[i];
      MS_EXCEPTION(ValueError) << "For " + op_name + "slice shape can't bigger than origin shape " +
                                    std::to_string(x_shape[i]) + "," + std::to_string(y);
    }
  }
  return std::make_shared<abstract::AbstractTensor>(data_type, size);
}
REGISTER_PRIMITIVE_C(kNameSliceFusion, SliceFusion);
}  // namespace ops
}  // namespace mindspore

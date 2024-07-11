/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ops/roi_pooling.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void ROIPooling::set_pooled_h(const int64_t pooled_h) { (void)this->AddAttr(kPooledH, MakeValue(pooled_h)); }

int64_t ROIPooling::get_pooled_h() const { return GetValue<int64_t>(GetAttr(kPooledH)); }

void ROIPooling::set_pooled_w(const int64_t pooled_w) { (void)this->AddAttr(kPooledW, MakeValue(pooled_w)); }

int64_t ROIPooling::get_pooled_w() const {
  auto value_ptr = GetAttr(kPooledW);
  return GetValue<int64_t>(value_ptr);
}

void ROIPooling::set_scale(const float scale) { (void)this->AddAttr(kScale, MakeValue(scale)); }

float ROIPooling::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<float>(value_ptr);
}

void ROIPooling::Init(const int64_t pooled_h, const int64_t pooled_w, const float scale) {
  this->set_pooled_h(pooled_h);
  this->set_pooled_w(pooled_w);
  this->set_scale(scale);
}
AbstractBasePtr ROIPoolingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);

  // Infer type
  auto output_data_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();

  // Infer shape
  auto new_h = GetValue<int64_t>(primitive->GetAttr(kPooledH));
  auto new_w = GetValue<int64_t>(primitive->GetAttr(kPooledW));
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto roi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  std::vector<int64_t> output_shape;
  output_shape.push_back(roi_shape[0]);
  output_shape.push_back(new_h);
  output_shape.push_back(new_w);
  output_shape.push_back(input_shape[1]);

  return std::make_shared<abstract::AbstractTensor>(output_data_type, std::make_shared<abstract::Shape>(output_shape));
}
REGISTER_PRIMITIVE_C(kNameROIPooling, ROIPooling);
}  // namespace ops
}  // namespace mindspore

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

#include <memory>
#include "ops/prior_box.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void PriorBox::set_min_sizes(const std::vector<int64_t> &min_sizes) {
  (void)this->AddAttr(kMinSizes, MakeValue(min_sizes));
}

std::vector<int64_t> PriorBox::get_min_sizes() const { return GetValue<std::vector<int64_t>>(GetAttr(kMinSizes)); }

void PriorBox::set_max_sizes(const std::vector<int64_t> &max_sizes) {
  (void)this->AddAttr(kMaxSizes, MakeValue(max_sizes));
}

std::vector<int64_t> PriorBox::get_max_sizes() const {
  auto value_ptr = GetAttr(kMaxSizes);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PriorBox::set_aspect_ratios(const std::vector<float> &aspect_ratios) {
  (void)this->AddAttr(kAspectRatios, MakeValue(aspect_ratios));
}

std::vector<float> PriorBox::get_aspect_ratios() const { return GetValue<std::vector<float>>(GetAttr(kAspectRatios)); }

void PriorBox::set_variances(const std::vector<float> &variances) {
  (void)this->AddAttr(kVariances, MakeValue(variances));
}

std::vector<float> PriorBox::get_variances() const {
  auto value_ptr = GetAttr(kVariances);
  return GetValue<std::vector<float>>(value_ptr);
}

void PriorBox::set_image_size_w(const int64_t image_size_w) {
  (void)this->AddAttr(kImageSizeW, MakeValue(image_size_w));
}

int64_t PriorBox::get_image_size_w() const {
  auto value_ptr = GetAttr(kImageSizeW);
  return GetValue<int64_t>(value_ptr);
}

void PriorBox::set_image_size_h(const int64_t image_size_h) {
  (void)this->AddAttr(kImageSizeH, MakeValue(image_size_h));
}

int64_t PriorBox::get_image_size_h() const {
  auto value_ptr = GetAttr(kImageSizeH);
  return GetValue<int64_t>(value_ptr);
}

void PriorBox::set_step_w(const float step_w) { (void)this->AddAttr(kStepW, MakeValue(step_w)); }

float PriorBox::get_step_w() const {
  auto value_ptr = GetAttr(kStepW);
  return GetValue<float>(value_ptr);
}

void PriorBox::set_step_h(const float step_h) { (void)this->AddAttr(kStepH, MakeValue(step_h)); }

float PriorBox::get_step_h() const {
  auto value_ptr = GetAttr(kStepH);
  return GetValue<float>(value_ptr);
}

void PriorBox::set_clip(const bool clip) { (void)this->AddAttr(kClip, MakeValue(clip)); }

bool PriorBox::get_clip() const {
  auto value_ptr = GetAttr(kClip);
  return GetValue<bool>(value_ptr);
}

void PriorBox::set_flip(const bool flip) { (void)this->AddAttr(kFlip, MakeValue(flip)); }

bool PriorBox::get_flip() const { return GetValue<bool>(GetAttr(kFlip)); }

void PriorBox::set_offset(const float offset) { (void)this->AddAttr(kOffset, MakeValue(offset)); }

float PriorBox::get_offset() const {
  auto value_ptr = GetAttr(kOffset);
  return GetValue<float>(value_ptr);
}

void PriorBox::Init(const std::vector<int64_t> &min_sizes, const std::vector<int64_t> &max_sizes,
                    const std::vector<float> &aspect_ratios, const std::vector<float> &variances,
                    const int64_t image_size_w, const int64_t image_size_h, const float step_w, const float step_h,
                    const bool clip, const bool flip, const float offset) {
  this->set_min_sizes(min_sizes);
  this->set_max_sizes(max_sizes);
  this->set_aspect_ratios(aspect_ratios);
  this->set_variances(variances);
  this->set_image_size_w(image_size_w);
  this->set_image_size_h(image_size_h);
  this->set_step_w(step_w);
  this->set_step_h(step_h);
  this->set_clip(clip);
  this->set_flip(flip);
  this->set_offset(offset);
}

AbstractBasePtr PriorBoxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = GetValue<std::vector<float>>(primitive->GetAttr(kAspectRatios));
  for (size_t i = 0; i < aspect_ratios.size(); i++) {
    float ratio = aspect_ratios[i];
    bool exist = std::any_of(different_aspect_ratios.begin(), different_aspect_ratios.end(),
                             [&](float v) { return abs(ratio - v) < 1e-6; });
    if (!exist) {
      (void)different_aspect_ratios.emplace_back(ratio);
      if (GetValue<bool>(primitive->GetAttr(kFlip))) {
        (void)different_aspect_ratios.emplace_back(1.0f / ratio);
      }
    }
  }
  auto min_sizes = GetValue<std::vector<int64_t>>(primitive->GetAttr(kMinSizes));
  int64_t num_priors_box = SizeToLong(min_sizes.size() * different_aspect_ratios.size() + min_sizes.size());
  auto input = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  int64_t h = input[0] * input[1] * num_priors_box * 4;
  std::vector<int64_t> output_shape{1, h, 1, 2};
  return std::make_shared<abstract::AbstractTensor>(kFloat32, output_shape);
}
REGISTER_PRIMITIVE_C(kNamePriorBox, PriorBox);
}  // namespace ops
}  // namespace mindspore

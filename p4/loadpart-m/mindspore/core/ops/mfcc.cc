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

#include "ops/mfcc.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input0_size = 3;
  const int64_t input1_size = 1;
  auto first_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto second_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input 0 rank", SizeToLong(first_input_shape.size()), kEqual, input0_size,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input 1 rank", SizeToLong(second_input_shape.size()), kEqual, input1_size,
                                           prim_name);
  std::vector<int64_t> out_shape = {first_input_shape[0], first_input_shape[1],
                                    GetValue<int64_t>(primitive->GetAttr(kDctCoeffNum))};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return infer_type;
}
}  // namespace

void Mfcc::Init(const float freq_upper_limit, const float freq_lower_limit, const int64_t filter_bank_channel_num,
                const int64_t dct_coeff_num) {
  this->set_freq_upper_limit(freq_upper_limit);
  this->set_freq_lower_limit(freq_lower_limit);
  this->set_filter_bank_channel_num(filter_bank_channel_num);
  this->set_dct_coeff_num(dct_coeff_num);
}

void Mfcc::set_freq_upper_limit(const float freq_upper_limit) {
  (void)this->AddAttr(kFreqUpperLimit, MakeValue(freq_upper_limit));
}

float Mfcc::get_freq_upper_limit() const {
  auto value_ptr = this->GetAttr(kFreqUpperLimit);
  return GetValue<float>(value_ptr);
}

void Mfcc::set_freq_lower_limit(const float freq_lower_limit) {
  (void)this->AddAttr(kFreqLowerLimit, MakeValue(freq_lower_limit));
}

float Mfcc::get_freq_lower_limit() const {
  auto value_ptr = this->GetAttr(kFreqLowerLimit);
  return GetValue<float>(value_ptr);
}

void Mfcc::set_filter_bank_channel_num(const int64_t filter_bank_channel_num) {
  (void)this->AddAttr(kFilterBankChannelNum, MakeValue(filter_bank_channel_num));
}

int64_t Mfcc::get_filter_bank_channel_num() const {
  auto value_ptr = this->GetAttr(kFilterBankChannelNum);
  return GetValue<int64_t>(value_ptr);
}

void Mfcc::set_dct_coeff_num(const int64_t dct_coeff_num) {
  (void)this->AddAttr(kDctCoeffNum, MakeValue(dct_coeff_num));
}

int64_t Mfcc::get_dct_coeff_num() const { return GetValue<int64_t>(GetAttr(kDctCoeffNum)); }

AbstractBasePtr MfccInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(input_args), InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameMfcc, Mfcc);
}  // namespace ops
}  // namespace mindspore

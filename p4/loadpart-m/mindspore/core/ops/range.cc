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
#include "ops/range.h"
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
void Range::set_d_type(const int64_t d_type) { (void)this->AddAttr(kDType, MakeValue(d_type)); }

int64_t Range::get_d_type() const {
  auto value_ptr = GetAttr(kDType);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_start(const int64_t start) { (void)this->AddAttr(kStart, MakeValue(start)); }

int64_t Range::get_start() const { return GetValue<int64_t>(GetAttr(kStart)); }

void Range::set_limit(const int64_t limit) { (void)this->AddAttr(kLimit, MakeValue(limit)); }

int64_t Range::get_limit() const {
  auto value_ptr = GetAttr(kLimit);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_delta(const int64_t delta) { (void)this->AddAttr(kDelta, MakeValue(delta)); }

int64_t Range::get_delta() const {
  auto value_ptr = GetAttr(kDelta);
  return GetValue<int64_t>(value_ptr);
}

void Range::Init(const int64_t d_type, const int64_t start, const int64_t limit, const int64_t delta) {
  this->set_d_type(d_type);
  this->set_start(start);
  this->set_limit(limit);
  this->set_delta(delta);
}

AbstractBasePtr RangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  int64_t shape_size = 0;
  const size_t max_input_num = 3;
  if (input_args.size() == max_input_num) {
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]->BuildValue());
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]->BuildValue());
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]->BuildValue());
    auto start_tensor = input_args[kInputIndex0]->BuildValue()->cast<tensor::TensorPtr>();
    auto limit_tensor = input_args[kInputIndex1]->BuildValue()->cast<tensor::TensorPtr>();
    auto delta_tensor = input_args[kInputIndex2]->BuildValue()->cast<tensor::TensorPtr>();
    auto dtype = start_tensor->data_type();
    switch (dtype) {
      case kNumberTypeInt:
      case kNumberTypeInt32: {
        auto start = *reinterpret_cast<int *>(start_tensor->data_c());
        auto limit = *reinterpret_cast<int *>(limit_tensor->data_c());
        auto delta = *reinterpret_cast<int *>(delta_tensor->data_c());
        shape_size =
          std::max(static_cast<int64_t>(std::ceil(static_cast<float>(limit - start) / delta)), static_cast<int64_t>(0));
      } break;
      case kNumberTypeFloat32:
      case kNumberTypeFloat: {
        auto start = *reinterpret_cast<float *>(start_tensor->data_c());
        auto limit = *reinterpret_cast<float *>(limit_tensor->data_c());
        auto delta = *reinterpret_cast<float *>(delta_tensor->data_c());
        shape_size =
          std::max(static_cast<int64_t>(std::ceil(static_cast<float>(limit - start) / delta)), static_cast<int64_t>(0));
      } break;
      default: {
        MS_LOG(EXCEPTION) << "Range has unsupported dataType: " << dtype;
      }
    }
  } else {
    int64_t start = GetValue<int64_t>(primitive->GetAttr(kStart));
    int64_t limit = GetValue<int64_t>(primitive->GetAttr(kLimit));
    int64_t delta = GetValue<int64_t>(primitive->GetAttr(kDelta));
    shape_size =
      std::max(static_cast<int64_t>(std::ceil(LongToDouble(limit - start) / delta)), static_cast<int64_t>(0));
  }
  return std::make_shared<abstract::AbstractTensor>(
    kInt32, std::make_shared<abstract::Shape>(std::vector<int64_t>{shape_size}));
}
REGISTER_PRIMITIVE_C(kNameRange, Range);
}  // namespace ops
}  // namespace mindspore

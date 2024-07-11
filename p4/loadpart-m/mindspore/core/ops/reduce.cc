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
#include "ops/reduce.h"
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
namespace {
void reduce_one_axis(const int64_t one_axis, const int64_t dim, std::set<int64_t> axis_reduce) {
  CheckAndConvertUtils::CheckInRange("axis", one_axis, kIncludeLeft, {-dim, dim}, "Reduce");
  if (one_axis < 0) {
    axis_reduce.insert(one_axis);
  }
}

std::vector<int64_t> infer_shape_reduce(std::vector<int64_t> input_x_shape, const ValuePtr axis_value,
                                        const bool keep_dims) {
  int64_t dim = SizeToLong(input_x_shape.size());
  std::set<int64_t> axis_reduce;
  if (axis_value == nullptr) {
    std::vector<int64_t> vec;
    if (keep_dims) {
      return std::vector<int64_t>(dim, 1);
    }
    return vec;
  }
  auto axis_value_elem = GetValue<std::vector<int64_t>>(axis_value);
  if (axis_value_elem.size() == 1) {
    reduce_one_axis(axis_value_elem[0], dim, axis_reduce);
  } else {
    size_t size = axis_value_elem.size();
    for (size_t i = 0; i < size; i++) {
      reduce_one_axis(axis_value_elem[i], dim, axis_reduce);
    }
  }
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < dim; i++) {
    if (axis_reduce.find(i) != axis_reduce.end()) {
      if (keep_dims) {
        (void)out_shape.emplace_back(1);
      }
    } else {
      (void)out_shape.emplace_back(input_x_shape[LongToSize(i)]);
    }
  }
  return out_shape;
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto axis_value = input_args[1]->BuildValue();

  MS_EXCEPTION_IF_NULL(primitive);
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];

  auto keep_dims = GetValue<bool>(primitive->GetAttr(kKeepDims));
  auto out_shape = infer_shape_reduce(input_x_shape, axis_value, keep_dims);

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), common_valid_types,
                                                    prim->name());
}
}  // namespace

void Reduce::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, MakeValue(keep_dims)); }

bool Reduce::get_keep_dims() const { return GetValue<bool>(GetAttr(kKeepDims)); }

void Reduce::Init(const bool keep_dims) { this->set_keep_dims(keep_dims); }

AbstractBasePtr ReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameReduce, Reduce);
}  // namespace ops
}  // namespace mindspore

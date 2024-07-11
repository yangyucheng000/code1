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

#include "ops/pack.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> _get_pack_shape(std::vector<BaseShapePtr> x_shapes, std::vector<TypePtr> x_types, int64_t axis,
                                     const std::string &name) {
  (void)CheckAndConvertUtils::CheckInteger("len of input_x", (int64_t)x_shapes.size(), kGreaterEqual, 1, name);
  (void)CheckAndConvertUtils::CheckSubClass("input_x[0]", x_types[0], {TypeIdToType(kObjectTypeTensorType)}, name);
  auto output_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shapes[0])[kShape];
  int64_t rank_base = SizeToLong(output_shape.size());
  int64_t N = SizeToLong(x_shapes.size());
  if (axis < 0) {
    axis = axis + rank_base + 1;
  }
  for (int64_t i = 1; i < N; i++) {
    auto type = x_types[LongToSize(i)]->cast<TensorTypePtr>()->element();
    MS_EXCEPTION_IF_NULL(type);
    auto type0 = x_types[0]->cast<TensorTypePtr>()->element();
    MS_EXCEPTION_IF_NULL(type0);
    CheckAndConvertUtils::Check("x_type[" + std::to_string(i) + "]", type->type_id(), kEqual, type0->type_id(), name);
    auto shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shapes[LongToSize(i)])[kShape];
    if (shape != output_shape) {
      MS_EXCEPTION(ValueError) << "For '" + name + "' element " + std::to_string(i) +
                                    "shape in input can't pack with first element.";
    }
  }
  (void)output_shape.insert(output_shape.begin() + axis, N);
  return output_shape;
}
}  // namespace

void Pack::set_axis(const int64_t &axis) { (void)AddAttr(kAxis, MakeValue(axis)); }

int64_t Pack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Pack::Init(const int64_t &axis) { this->set_axis(axis); }

AbstractBasePtr PackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_shapes = input_args[0]->BuildShape()->cast<abstract::TupleShapePtr>()->shape();
  auto x_types = input_args[0]->BuildType()->cast<TuplePtr>()->elements();
  auto all_shape = _get_pack_shape(x_shapes, x_types, GetValue<int64_t>(primitive->GetAttr(kAxis)), prim_name);
  auto tensor_type = x_types[0]->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return std::make_shared<abstract::AbstractTensor>(data_type, all_shape);
}
REGISTER_PRIMITIVE_C(kNamePack, Pack);
}  // namespace ops
}  // namespace mindspore

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
#include "ops/hashtable_lookup.h"

#include <vector>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr HashtableLookupInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);
  std::vector<int64_t> hits_shape;
  auto input = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("logits size", SizeToLong(input.size()), kGreaterEqual, 1, op_name);
  hits_shape.push_back(input[0]);

  auto value_type = input_args[kInputIndex2]->BuildType();
  MS_EXCEPTION_IF_NULL(value_type);
  auto tensor_type = value_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  std::vector<int64_t> value_shape;
  auto output = std::make_shared<abstract::AbstractTensor>(data_type, value_shape);
  auto hits = std::make_shared<abstract::AbstractTensor>(kInt8, hits_shape);
  AbstractBasePtrList output1 = {output, hits};
  return std::make_shared<abstract::AbstractTuple>(output1);
}
REGISTER_PRIMITIVE_C(kNameHashtableLookup, HashtableLookup);
}  // namespace ops
}  // namespace mindspore

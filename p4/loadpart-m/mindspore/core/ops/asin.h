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

#ifndef MINDSPORE_CORE_OPS_ASIN_H_
#define MINDSPORE_CORE_OPS_ASIN_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAsin = "Asin";
/// \brief Computes arcsine of input tensors element-wise.
/// Refer to Python API @ref mindspore.ops.Asin for more details.
class MS_CORE_API Asin : public PrimitiveC {
 public:
  /// \brief Constructor.
  Asin() : PrimitiveC(kNameAsin) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~Asin() = default;
  MS_DECLARE_PARENT(Asin, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Asin for the inputs.
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ASIN_H_
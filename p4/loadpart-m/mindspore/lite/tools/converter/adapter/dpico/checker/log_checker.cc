/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/log_checker.h"
#include <vector>
#include <string>
#include <limits>
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
bool LogChecker::Check(CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, kInputIndex1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }

  auto primitive = GetValueNode<PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto base_ptr = primitive->GetAttr(ops::kBase);
  if (base_ptr != nullptr) {
    auto base_data = GetValue<float>(base_ptr);  // support -1.0 && any positive num but 1.0
    if (fabs(base_data + 1.0) <= std::numeric_limits<float>::epsilon() ||
        (base_data > 0 && fabs(base_data - 1.0) > std::numeric_limits<float>::epsilon())) {
      return true;
    } else {
      MS_LOG(WARNING) << "base val only supports -1.0 or any positive num but 1.0 " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}
OpCheckerRegistrar g_LogChecker("Log", new LogChecker());
}  // namespace dpico
}  // namespace mindspore

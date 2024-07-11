/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/dynamic_shape/link_custom_op.h"

#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/ascend/dynamic_shape/ascend_dynamic_shape_helper.h"

namespace mindspore {
namespace opt::dynamic_shape {
namespace {
constexpr size_t kTupleFirstItemIndex = 0;
constexpr size_t kFirstDataInputIndex = 1;

AnfNodePtr InsertDepend(const FuncGraphPtr &g, const AnfNodePtr &prev, const AnfNodePtr &next) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(next);
  // add depend from prev to next
  auto depend_node = g->NewCNode(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), next, prev});
  MS_EXCEPTION_IF_NULL(depend_node);
  return depend_node;
}

bool LinkInternalOp(const FuncGraphPtr &g, const AnfNodePtr &node, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(depend_nodes);
  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(node);
  if (custom_nodes.infer_node != nullptr) {
    if (custom_nodes.init_node == nullptr) {
      MS_LOG(WARNING) << "Node " << node->DebugString() << " has infer node but init node is null.";
    } else {
      depend_nodes->push_back(InsertDepend(g, custom_nodes.infer_node, custom_nodes.init_node));  // link infer => init
      depend_nodes->push_back(InsertDepend(g, custom_nodes.init_node, node));                     // link init => launch
      changed = true;
    }
  }

  if (IsDynUpdate(custom_nodes.update_node)) {
    depend_nodes->push_back(InsertDepend(g, node, custom_nodes.update_node));  // link launch => update
    changed = true;
  }

  return changed;
}

bool LinkInputOp(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_nodes);
  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(cnode);
  if (custom_nodes.infer_node == nullptr) {
    return changed;
  }
  size_t input_num = AnfAlgo::GetInputNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto prev = AnfAlgo::GetPrevNodeOutput(cnode, i);
    const auto &prev_node = prev.first;
    if (prev_node == nullptr || !CustomActorNodeManager::Instance().IsRegistered(prev_node)) {
      continue;
    }
    auto prev_custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(prev_node);
    if (prev_custom_nodes.infer_node != nullptr) {
      depend_nodes->push_back(
        InsertDepend(g, prev_custom_nodes.infer_node, custom_nodes.infer_node));  // link prev.infer => curr.infer
      MS_LOG(DEBUG) << "Link from " << prev_node->fullname_with_scope() << " infer "
                    << prev_custom_nodes.infer_node->fullname_with_scope() << " to " << cnode->fullname_with_scope()
                    << " infer " << custom_nodes.infer_node->fullname_with_scope();
      changed = true;
    }
    if (IsDynUpdate(prev_custom_nodes.update_node)) {
      depend_nodes->push_back(
        InsertDepend(g, prev_custom_nodes.update_node, custom_nodes.infer_node));  // link prev.update => curr.infer
      MS_LOG(DEBUG) << "Link from " << prev_node->fullname_with_scope() << " update "
                    << prev_custom_nodes.update_node->fullname_with_scope() << " to " << cnode->fullname_with_scope()
                    << " infer " << custom_nodes.infer_node->fullname_with_scope();
      changed = true;
    }
  }
  return changed;
}

bool LinkDependSync(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes) {
  MS_EXCEPTION_IF_NULL(g);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_nodes);
  bool changed = false;
  auto custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(cnode);
  if (custom_nodes.infer_node == nullptr) {
    return changed;
  }

  auto dynamic_shape_depends = abstract::GetDependsFormMap(cnode);
  if (dynamic_shape_depends.empty()) {
    return changed;
  }

  for (auto depend_index : dynamic_shape_depends) {
    auto prev = AnfAlgo::GetPrevNodeOutput(cnode, depend_index);
    const auto &prev_node = prev.first;
    if (prev_node == nullptr || !CustomActorNodeManager::Instance().IsRegistered(prev_node)) {
      continue;
    }

    // If previous node is dynamic, so it was already link.
    auto prev_custom_nodes = CustomActorNodeManager::Instance().GetCustomActorNodes(prev_node);
    if (IsDynUpdate(prev_custom_nodes.update_node)) {
      continue;
    }

    // 1. Link prev_node => prev_node.update if its update is just sync.
    depend_nodes->push_back(InsertDepend(g, prev_node, prev_custom_nodes.update_node));
    // 2. Link prev_node.update => cur_node.infer.
    depend_nodes->push_back(InsertDepend(g, prev_custom_nodes.update_node, custom_nodes.infer_node));
    changed = true;
  }
  return changed;
}

/**
 * @brief Attach Custom's Depend nodes with additional MakeTuple and TupleGetItem before graph return.
 *
 *          %0 = A
 *          return %0
 *          ---->
 *          %0 = A
 *          %1 = MakeTuple(%0, %depend0, %depend1...)
 *          %2 = TupleGetItem(%1, 0)
 *          return %2
 *
 * @param g Graph.
 * @param depend_nodes Custom's Depend nodes.
 */
void AttachDependNodes(const FuncGraphPtr &g, const AnfNodePtrList &depend_nodes) {
  if (depend_nodes.empty()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(g);
  auto return_node = g->get_return();
  MS_EXCEPTION_IF_NULL(return_node);

  // New MakeTuple node
  auto mk_inputs = AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())),
                                  return_node->input(kFirstDataInputIndex)};
  mk_inputs.insert(mk_inputs.end(), depend_nodes.begin(), depend_nodes.end());
  auto make_tuple_node = g->NewCNode(mk_inputs);

  // Get first element item form that maketuple and return.
  auto get_1st_item = g->NewCNode(AnfNodePtrList{NewValueNode(std::make_shared<Primitive>(prim::kTupleGetItem)),
                                                 make_tuple_node, NewValueNode(SizeToLong(kTupleFirstItemIndex))});

  // Attach back.
  return_node->set_input(kFirstDataInputIndex, get_1st_item);
}
}  // namespace

bool LinkCustomOp::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  AnfNodePtrList depend_nodes;
  auto node_list = TopoSort(func_graph->get_return());
  for (const auto &node : node_list) {
    CNodePtr cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !CustomActorNodeManager::Instance().IsRegistered(cnode)) {
      continue;
    }

    changed = LinkInternalOp(func_graph, cnode, &depend_nodes) || changed;
    changed = LinkInputOp(func_graph, cnode, &depend_nodes) || changed;
    changed = LinkDependSync(func_graph, cnode, &depend_nodes) || changed;
  }

  CustomActorNodeManager::Instance().Reset();

  if (changed) {
    AttachDependNodes(func_graph, depend_nodes);

    // Rebuild graph's edge.
    auto mng = func_graph->manager();
    if (mng == nullptr) {
      mng = Manage(func_graph, true);
      func_graph->set_manager(mng);
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore

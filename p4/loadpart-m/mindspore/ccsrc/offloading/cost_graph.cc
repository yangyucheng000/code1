#include "cost_graph.h"

namespace mindspore {

namespace offloading {

static const std::unordered_set<std::string> fused_cnodes_set = { kFusedMatMulBiasAddName };

// assume fused nodes are connected linearly
static const std::unordered_map<std::string, std::vector<std::string>> fused_equ_name_map = { 
  { kFusedMatMulBiasAddName, { prim::kPrimMatMul->name(), prim::kPrimBiasAdd->name() } }
};

static const PrimitiveSet forbidden_cnodes = { prim::kPrimMakeTuple,   prim::kPrimStateSetItem, //prim::kPrimTupleGetItem,
                                               prim::kPrimReturn,      prim::kPrimPartial,      prim::kPrimDepend,
                                               prim::kPrimUpdateState, prim::kPrimLoad,         prim::kPrimBatchNorm };

static const std::unordered_set<std::string> allowed_cps_node_names = { "ReLU", "MaxPool", "Concat", "Pad" };

std::unordered_map<std::string, float> LoadExecTimeTSV(const std::string &path, size_t &bsz) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    MS_LOG(EXCEPTION) << "Open file '" << path << "' failed!";
  }
  std::string line;
  std::string name;
  float time;
  std::unordered_map<std::string, float> new_time_map;
  new_time_map.clear();
  ifs >> bsz;
  while (std::getline(ifs, line)) {
      std::stringstream ss(line);
      ss >> name >> time;
      if (name == "") continue;
      new_time_map[name] = time;
  }
  ifs.close();
  return new_time_map;
}

void DumpExecTimeTSV(const std::string &path, std::unordered_map<std::string, float>& time_map, size_t bsz) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(EXCEPTION) << "Open file '" << path << "' failed!";
  }
  ofs << bsz << std::endl;
  for (auto &p : time_map) {
    ofs << p.first << '\t' << p.second << std::endl;
  }
  ofs.close();
}

std::string GetCutLabel(const std::unordered_set<CostGraph::NodePtr> &cut_nodes) {
  std::vector<std::string> cut_node_names;
  cut_node_names.reserve(cut_nodes.size());
  for (const auto& n : cut_nodes) {
    cut_node_names.push_back(n->name_);
  }
  std::sort(cut_node_names.begin(), cut_node_names.end());

  std::string res = std::accumulate(cut_node_names.begin(), cut_node_names.end(), std::string(), [](const std::string &acc, const std::string &name) {
    return acc.empty() ? name : acc + "," + name;
  });

  return res;
}

std::string GetCutLabel(std::vector<std::string> &cut_node_names) {
  std::sort(cut_node_names.begin(), cut_node_names.end());

  std::string res = std::accumulate(cut_node_names.begin(), cut_node_names.end(), std::string(), [](const std::string &acc, const std::string &name) {
    return acc.empty() ? name : acc + "," + name;
  });

  return res;
}

std::unordered_map<std::string, float> GenerateTimeMapWithRenaming(KernelGraphPtr &g, std::unordered_map<std::string, float> &graph_profile) {
  MS_EXCEPTION_IF_NULL(g);
  std::unordered_map<std::string, int> rename_count_map;
  std::unordered_map<std::string, float> time_map;

  auto vec = TopoSort(g->get_return());
  for (auto &n : vec) {
    if (!n->isa<CNode>()) {
      continue;
    }

    auto cnode = n->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!AnfUtils::IsRealKernel(cnode)) {
      continue;
    }

    if (fused_cnodes_set.find(AnfAlgo::GetCNodeName(cnode)) == fused_cnodes_set.end()) {
      auto prim_name = AnfAlgo::GetCNodePrimitive(cnode)->name();
      auto it = rename_count_map.find(prim_name);
      if (it == rename_count_map.end()) {
        rename_count_map[prim_name] = 0;
      } else {
        rename_count_map[prim_name] += 1;
      }
      std::string new_name = prim_name + "-" + std::to_string(rename_count_map[prim_name]);
      if (!opt::IsNopNode(cnode)) {
        auto it = graph_profile.find(cnode->fullname_with_scope());
        if (it == graph_profile.end()) {
          MS_LOG(EXCEPTION) << "Cannot find profile info of " << cnode->fullname_with_scope();
        }
        time_map[new_name] = it->second;
      } else {
        time_map[new_name] = 0.0;
      }
    } else {
      auto fused_prim_name = AnfAlgo::GetCNodePrimitive(cnode)->name();
      auto name_it = fused_equ_name_map.find(fused_prim_name);
      if (name_it == fused_equ_name_map.end()) {
        MS_LOG(EXCEPTION) << "Handler for fused CNode " << fused_prim_name << " not created";
      }

      auto &name_list = name_it->second;
      for (size_t idx = 0; idx < name_list.size(); ++idx) {
        // rename
        auto &prim_name = name_list[idx];
        auto it = rename_count_map.find(prim_name);
        if (it == rename_count_map.end()) {
          rename_count_map[prim_name] = 0;
        } else {
          rename_count_map[prim_name] += 1;
        }
        std::string new_name = prim_name + "-" + std::to_string(rename_count_map[prim_name]);
        if (idx == 0) {
          if (!opt::IsNopNode(cnode)) {
            auto it = graph_profile.find(cnode->fullname_with_scope());
            if (it == graph_profile.end()) {
              MS_LOG(EXCEPTION) << "Cannot find profile info of " << cnode->fullname_with_scope();
            }
            time_map[new_name]= it->second;
          } else {
            time_map[new_name] = 0.0;
          }
        } else {
          time_map[new_name] = 0.0;
        }
      }
    }
  }
  return time_map;
}

CostGraph::CostGraph(KernelGraphPtr graph, std::string graph_profile_path, double scale_factor) 
  : graph_(graph)
{ 
  GraphProfile profile_proto;
  std::ifstream ifs(graph_profile_path, std::ios::binary);
  if (!profile_proto.ParseFromIstream(&ifs)) {
    MS_LOG(EXCEPTION) << "CostGraph: failed to load GraphProfile at " << graph_profile_path;
  }
  ifs.close();
  for (const auto& batch_entry : profile_proto.entries()) {
    // use profile of batch size = 1 as local_exec_time_
    if (batch_entry.batch_size() == 1) {
      for (const auto& profile_entry : batch_entry.profile()) {
        graph_profile_[profile_entry.name()] = profile_entry.time() * scale_factor;
      }
      break;
    }
  }
  Construct();
}

void CostGraph::ConstructSingleNode(const CNodePtr &cnode, const NodeUsersMap &node_user_map) {
  // new Node
  auto node = std::make_shared<Node>();
  // rename
  auto prim_name = AnfAlgo::GetCNodePrimitive(cnode)->name();
  auto it = rename_count_map_.find(prim_name);
  if (it == rename_count_map_.end()) {
    rename_count_map_[prim_name] = 0;
  } else {
    rename_count_map_[prim_name] += 1;
  }
  std::string new_name = prim_name + "-" + std::to_string(rename_count_map_[prim_name]);
  node->name_ = new_name;
  // set real execution time
  if (!opt::IsNopNode(cnode)) {
    auto it = graph_profile_.find(cnode->fullname_with_scope());
    if (it == graph_profile_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find profile info of " << cnode->fullname_with_scope();
      return;
    }
    node->local_exec_time_ = it->second;
  } else {
    node->local_exec_time_ = 0.0;
  }

  // get input real nodes and set input nodes
  size_t input_num = AnfAlgo::GetInputNum(cnode);
  for (size_t i = 0; i < input_num; ++i) {
    auto input = AnfAlgo::GetInputNode(cnode, i);
    if (input->isa<CNode>()) {
      // deal with input CNodes
      if (!IsPrimitiveCNode(input, prim::kPrimLoad)) {
        auto it = real_fake_map_.find(input);
        if (it != real_fake_map_.end()) {
          node->inputs_.push_back(real_fake_map_[input].back());
        } else {
          MS_LOG(EXCEPTION) << "Corresponding Node of " << input->fullname_with_scope() << " not generated";
          return;
        }
        // get input sizes
        auto &input_node_output_size = node->inputs_.back()->output_sizes_;
        if (input_node_output_size.empty()) {
          MS_LOG(EXCEPTION) << "Output size of corresponding Node of " << input->fullname_with_scope() << " not populated";
          return;
        }
        node->input_sizes_.push_back(std::accumulate(input_node_output_size.begin(), input_node_output_size.end(), 0));
      } else { // directly add Load nodes
        node->real_cnodes_.push_back(input->cast<CNodePtr>());
      }
    }
  }
  // this cnode comes last
  node->real_cnodes_.push_back(cnode);
  node->output_sizes_.clear();
  // corner case: bind TupleGetItem with BatchNorm
  bool has_tuple_get_item = false;
  auto output_info_it = node_user_map.find(cnode);
  if (output_info_it == node_user_map.end()) {
    MS_LOG(EXCEPTION) << "CNode " << cnode->fullname_with_scope() << " is not a node of the full graph";
    return;
  }
  auto &output_info_list = output_info_it->second;
  AnfNodePtr tuple_get_item = nullptr;
  for (auto &out_node_info : output_info_list) {
    if (IsPrimitiveCNode(out_node_info.first, prim::kPrimTupleGetItem)) {
      has_tuple_get_item = true;
      tuple_get_item = out_node_info.first;
      node->real_cnodes_.push_back(tuple_get_item->cast<CNodePtr>());
      auto output_num = AnfAlgo::GetOutputTensorNum(tuple_get_item);
      for (size_t j = 0; j < output_num; ++j) {
        node->output_sizes_.push_back(AnfAlgo::GetOutputTensorMemSize(tuple_get_item, j));
      }
    }
  }
  // calculate output sizes
  if (!has_tuple_get_item) {
    auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t j = 0; j < output_num; ++j) {
      node->output_sizes_.push_back(AnfAlgo::GetOutputTensorMemSize(cnode, j));
    }
  }
  // add node to real_fake_map_
  real_fake_map_[cnode] = {node};
  if (tuple_get_item) {
    real_fake_map_[tuple_get_item] = {node};
  }
  local_time_map_[node->name_] = node->local_exec_time_;
  node_set_.insert(node);
}

void CostGraph::ConstructFusedNode(const CNodePtr &cnode, const NodeUsersMap &node_user_map) {
  auto fused_prim_name = AnfAlgo::GetCNodePrimitive(cnode)->name();
  auto name_it = fused_equ_name_map.find(fused_prim_name);
  if (name_it == fused_equ_name_map.end()) {
    MS_LOG(EXCEPTION) << "Handler for fused CNode " << fused_prim_name << " not created";
    return;
  }
  // new nodes
  auto &name_list = name_it->second;
  NodePtr last_node = nullptr;
  std::vector<NodePtr> tmp_nodes;
  for (size_t idx = 0; idx < name_list.size(); ++idx) {
    auto node = std::make_shared<Node>();
    // rename
    auto &prim_name = name_list[idx];
    auto it = rename_count_map_.find(prim_name);
    if (it == rename_count_map_.end()) {
      rename_count_map_[prim_name] = 0;
    } else {
      rename_count_map_[prim_name] += 1;
    }
    std::string new_name = prim_name + "-" + std::to_string(rename_count_map_[prim_name]);
    node->name_ = new_name;
    // set execution time for the first CNode, others are 0
    // get input real nodes and set input nodes
    if (idx == 0) {
      if (!opt::IsNopNode(cnode)) {
        auto it = graph_profile_.find(cnode->fullname_with_scope());
        if (it == graph_profile_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find profile info of " << cnode->fullname_with_scope();
          return;
        }
        node->local_exec_time_ = it->second;
      } else {
        node->local_exec_time_ = 0.0;
      }
      size_t input_num = AnfAlgo::GetInputNum(cnode);
      for (size_t i = 0; i < input_num; ++i) {
        auto input = AnfAlgo::GetInputNode(cnode, i);
        if (input->isa<CNode>()) {
          // deal with input CNodes
          if (!IsPrimitiveCNode(input, prim::kPrimLoad)) {
            auto it = real_fake_map_.find(input);
            if (it != real_fake_map_.end()) {
              node->inputs_.push_back(real_fake_map_[input].back());
            } else {
              MS_LOG(EXCEPTION) << "Corresponding Node of " << input->fullname_with_scope() << " not generated";
              return;
            }
            // get input sizes
            auto &input_node_output_size = node->inputs_.back()->output_sizes_;
            if (input_node_output_size.empty()) {
              MS_LOG(EXCEPTION) << "Output size of corresponding Node of " << input->fullname_with_scope() << " not populated";
              return;
            }
            node->input_sizes_.push_back(std::accumulate(input_node_output_size.begin(), input_node_output_size.end(), 0));
          } else { // directly add Load nodes
            node->real_cnodes_.push_back(input->cast<CNodePtr>());
          }
        }
      }
      // this cnode comes last
      node->real_cnodes_.push_back(cnode);
      // calculate output size
      node->output_sizes_.clear();
      auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
      for (size_t j = 0; j < output_num; ++j) {
        node->output_sizes_.push_back(AnfAlgo::GetOutputTensorMemSize(cnode, j));
      }
    } else {
      node->local_exec_time_ = 0.0;
      node->inputs_.push_back(last_node);
      node->real_cnodes_.clear();
      std::copy(last_node->output_sizes_.begin(), last_node->output_sizes_.end(), std::back_inserter(node->input_sizes_));
      std::copy(last_node->output_sizes_.begin(), last_node->output_sizes_.end(), std::back_inserter(node->output_sizes_));
    }
    last_node = node;
    local_time_map_[node->name_] = node->local_exec_time_;
    node_set_.insert(node);
    tmp_nodes.push_back(node);
  }
  // deal with duplicated Load nodes of the fused node
  fused_node_load_node_map_[cnode] = CNodePtrList();
  for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
    if (IsPrimitiveCNode(cnode->input(idx), prim::kPrimLoad)) {
      auto load_cnode = cnode->input(idx)->cast<CNodePtr>();
      auto param = load_cnode->input(kLoadRealInput)->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      auto it = node_user_map.find(param);
      if (it == node_user_map.end()) {
        MS_LOG(EXCEPTION) << "Parameter " << param->fullname_with_scope() << " is not a parameter of the full graph";
        return;
      }
      auto &output_info_list = it->second;
      for (auto &info : output_info_list) {
        if (info.first != load_cnode) {
          fused_node_load_node_map_[cnode].push_back(info.first->cast<CNodePtr>());
        }
      }
    }
  }
  real_fake_map_[cnode] = tmp_nodes;
}

void CostGraph::Construct() {
  MS_EXCEPTION_IF_NULL(graph_);
  auto manager = graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto& node_user_map = manager->node_users();

  auto vec = TopoSort(graph_->get_return());
  for (auto &n : vec) {
    if (!n->isa<CNode>()) {
      continue;
    }

    auto cnode = n->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    topo_list_.push_back(cnode);
    if (!AnfUtils::IsRealKernel(cnode)) {
      continue;
    }

    if (fused_cnodes_set.find(AnfAlgo::GetCNodeName(cnode)) == fused_cnodes_set.end()) {
      ConstructSingleNode(cnode, node_user_map);
    } else {
      ConstructFusedNode(cnode, node_user_map);
    }
  }
  GenerateSourceSinkNodes(node_user_map);
  GenerateEdgeSet();
  // construct name -> node map
  for (auto &n : node_set_) {
    name_node_map_[n->name_] = n;
  }
}

void CostGraph::GenerateEdgeSet() {
  if (node_set_.empty()) {
    return;
  }
  edge_set_.clear();
  for (auto &node : node_set_) {
    edge_set_[node] = std::vector<EdgeEndPtr>();
  }
  for (auto &node : node_set_) {
    for (auto &in_node : node->inputs_) {
      auto weight = std::accumulate(in_node->output_sizes_.begin(), in_node->output_sizes_.end(), 0);
      edge_set_[in_node].emplace_back(std::make_shared<EdgeEnd>(node, weight));
    }
  }
}

void CostGraph::ConstructReverseEdges() {
  if (node_set_.empty()) {
    return;
  }
  r_edge_set_.clear();
  for (auto &node : node_set_) {
    r_edge_set_[node] = std::vector<EdgeEndPtr>();
  }
  for (auto &node : node_set_) {
    auto &edges = edge_set_[node];
    for (auto &e : edges) {
      r_edge_set_[e->end_node_].emplace_back(std::make_shared<EdgeEnd>(node, e->trans_size_));
    }
  }
}

void CostGraph::GenerateSourceSinkNodes(const NodeUsersMap &node_user_map) {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto& input_nodes = graph_->input_nodes();
  int input_param_num = 0;
  for (auto &in : input_nodes) {
    auto param = in->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (!param->has_default()) {
      input_param_num++;
      if (input_param_num > 1) {
        MS_LOG(EXCEPTION) << "Currently does not support the existence of multiple input parameters";
        return;
      }
      // new node
      auto source_node = std::make_shared<Node>();
      source_node->name_ = param->fullname_with_scope();
      source_node->local_exec_time_ = 0.0;
      source_node->real_cnodes_.clear();
      source_node->inputs_.clear();
      source_node->input_sizes_.clear();
      auto output_num = AnfAlgo::GetOutputTensorNum(param);
      for (size_t j = 0; j < output_num; ++j) {
        source_node->output_sizes_.push_back(AnfAlgo::GetOutputTensorMemSize(param, j));
      }
      // find its user and insert to user's inputs and input_sizes
      auto it = node_user_map.find(param);
      if (it == node_user_map.end()) {
        MS_LOG(EXCEPTION) << "Parameter " << param->fullname_with_scope() << " is not a parameter of the full graph";
        return;
      }
      auto &output_info_list = it->second;
      for (auto &info : output_info_list) {
        auto node_it = real_fake_map_.find(info.first);
        if (node_it == real_fake_map_.end()) {
          MS_LOG(EXCEPTION) << "The corresponding Node of " << info.first->fullname_with_scope() << " is not generated";
          return;
        }
        auto &user_node = node_it->second.front();
        MS_EXCEPTION_IF_NULL(user_node);
        user_node->inputs_.insert(user_node->inputs_.begin(), source_node);
        size_t output_size = std::accumulate(source_node->output_sizes_.begin(), source_node->output_sizes_.end(), 0);
        user_node->input_sizes_.insert(user_node->input_sizes_.begin(), output_size);
      }
      // add to node_set
      local_time_map_[source_node->name_] = source_node->local_exec_time_;
      node_set_.insert(source_node);
      input_node_set_.insert(source_node);
      source_node_ = source_node;
    }
  }
  const auto& output_nodes = graph_->outputs();
  if (output_nodes.size() != 1) {
    MS_LOG(EXCEPTION) << "The only output CNode of KernelGraph should be Depend or MakeTuple";
    return;
  }
  CNodePtr depend_node = output_nodes[0]->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_node);
  if (!IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
    depend_node = depend_node->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
  }
  AnfNodePtrList real_outputs;
  if (!IsPrimitiveCNode(depend_node->input(1), prim::kPrimMakeTuple)) {
    real_outputs.push_back(depend_node->input(1));
  } else {
    auto make_tuple = depend_node->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto &make_tuple_inputs = make_tuple->inputs();
    std::copy(make_tuple_inputs.begin() + 1, make_tuple_inputs.end(), std::back_inserter(real_outputs));
  }

  auto sink_node = std::make_shared<Node>();
  sink_node->name_ = "Return";
  sink_node->local_exec_time_ = 0.0;
  sink_node->real_cnodes_.clear();
  for (auto &out : real_outputs) {
    // add inputs and input_sizes
    auto it = real_fake_map_.find(out);
    if (it == real_fake_map_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find Node counterpart of output " << out->fullname_with_scope();
      return;
    }
    sink_node->inputs_.push_back(it->second.back());
    auto &out_size = it->second.back()->output_sizes_;
    sink_node->input_sizes_.push_back(std::accumulate(out_size.begin(), out_size.end(), 0));
  }
  sink_node->output_sizes_.push_back(0.0);
  local_time_map_[sink_node->name_] = sink_node->local_exec_time_;
  node_set_.insert(sink_node);
  sink_node_ = sink_node;
}

void CostGraph::GetCutNodes() {
  std::unordered_map<NodePtr, int> dfn, low;
  if (input_node_set_.size() != 1) {
    MS_LOG(EXCEPTION) << "CostGraph::GetCutNodes: QDMP only support DAG with 1 input node currently";
  }
  auto &input_node = *input_node_set_.begin();
  if (edge_set_[input_node].size() > 1) {
    MS_LOG(EXCEPTION) << "CostGraph::GetCutNodes: QDMP only support DAG with 1 input node which is used by 1 node currently";
  }
  if (r_edge_set_.empty()) {
    ConstructReverseEdges();
  }
  Tarjan(0, input_node, nullptr, dfn, low);

  if (cg_topo_list_.empty()) {
    TopoSortCostGraph(cg_topo_list_);
  }
  for (size_t i = 0; i < cg_topo_list_.size(); ++i) {
    if (cg_topo_list_[i]->is_cut_) {
      cut_node_list_.emplace_back(i);
    }
  }
}

void CostGraph::GetNodesByName(const std::vector<std::string> &node_names, std::unordered_set<NodePtr> &nodes) {
  for (auto &name : node_names) {
    auto it = name_node_map_.find(name);
    if (it == name_node_map_.end()) {
      MS_LOG(EXCEPTION) << "CostGraph::GetNodesByName: node with name " << name << " not found";
    }
    nodes.insert(it->second);
  }
}

void CostGraph::GetNodesByName(const std::unordered_map<std::string, tensor::TensorPtr> &input_map, std::unordered_set<NodePtr> &nodes) {
  for (auto &kv : input_map) {
    auto it = name_node_map_.find(kv.first);
    if (it == name_node_map_.end()) {
      MS_LOG(EXCEPTION) << "CostGraph::GetNodesByName: node with name " << kv.first << " not found";
    }
    nodes.insert(it->second);
  }
}

void CostGraph::Tarjan(int clock, const NodePtr &cur_node, const NodePtr &fa_node, std::unordered_map<NodePtr, int> &dfn, std::unordered_map<NodePtr, int> &low) {
  dfn[cur_node] = low[cur_node] = ++clock;
  for (auto &e : edge_set_[cur_node]) {
    auto &v = e->end_node_;
    if (dfn.find(v) == dfn.end()) {
      Tarjan(clock, v, cur_node, dfn, low);
      low[cur_node] = std::min(low[cur_node], low[v]);
      if (low[v] >= dfn[cur_node]) {
        cur_node->is_cut_ = true;
      }
    } else if (dfn[v] < dfn[cur_node] && v != fa_node) {
      low[cur_node] = std::min(low[cur_node], dfn[v]);
    }
  }
  for (auto &e : r_edge_set_[cur_node]) {
    auto &v = e->end_node_;
    if (dfn.find(v) == dfn.end()) {
      Tarjan(clock, v, cur_node, dfn, low);
      low[cur_node] = std::min(low[cur_node], low[v]);
      if (low[v] >= dfn[cur_node]) {
        cur_node->is_cut_ = true;
      }
    } else if (dfn[v] < dfn[cur_node] && v != fa_node) {
      low[cur_node] = std::min(low[cur_node], dfn[v]);
    }
  }
}

std::string CostGraph::GetUnifiedOutputName(const CNodePtr& cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto it = real_fake_map_.find(cnode);
  if (it == real_fake_map_.end()) {
    MS_LOG(EXCEPTION) << "Corresponding Node of " << cnode->fullname_with_scope() << " is not created";
    return "";
  }
  return it->second.back()->name_;
}

void CostGraph::DrawCostGraph(const std::string &path, bool is_local, bool print_shape) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }
  ofs << "digraph g {\n";
  ofs << "  fontname = \"Courier New\"\n";
  ofs << "  node [ fontname = \"Courier New\" ]\n";
  ofs << "  edge [ fontname = \"Courier New\" ]\n";
  ofs << "  graph [ fontsize = 24, spline = true, overlap = false ];\n";
  ofs << "  ratio = auto;\n";
  for (auto &node : node_set_) {
    double time = is_local ? node->local_exec_time_ / 1e3 : node->remote_exec_time_ / 1e3;
    if (node->is_cut_) {
      ofs << "  \"" << node->name_ << "\" [ label = \"" << node->name_ << "\n" << time << " ms\", color = \"0.650 0.200 1.000\", style = filled ];\n";
    } else {
      ofs << "  \"" << node->name_ << "\" [ label = \"" << node->name_ << "\n" << time << " ms\", color = \"0.650 0.200 1.000\" ];\n";
    }
  }
  for (auto &edge_list_p : edge_set_) {
    for (auto &edge : edge_list_p.second) {
      if (print_shape) {
        auto output_shape = GetOutputShape(edge_list_p.first);
        ofs << "  \"" << edge_list_p.first->name_ << "\" -> \"" << edge->end_node_->name_ << "\" [ label = \"";
        PrintVector(output_shape, ofs);
        ofs << "\" ];\n";
      } else {
        ofs << "  \"" << edge_list_p.first->name_ << "\" -> \"" << edge->end_node_->name_ << "\" [ label = " << edge->trans_size_ << " ];\n";
      }
    }
  }
  ofs << "}";
  ofs.close();
}

void CostGraph::CostGraphToCSV(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }
  if (cg_topo_list_.empty()) {
    TopoSortCostGraph(cg_topo_list_);
  }
  ofs << "op1,op2,trans,time1,time2\n";
  for (auto &node : cg_topo_list_) {
    for (auto &out_node : edge_set_[node]) {
      ofs << "\"" << node->name_ << "\",\"" 
          << out_node->end_node_->name_ << "\","
          << out_node->trans_size_ << ","
          << node->local_exec_time_ << ","
          << out_node->end_node_->local_exec_time_ << "\n";
    }
  }
  ofs.close();
}

void CostGraph::DumpTimeMapToFile(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }
  if (cg_topo_list_.empty()) {
    TopoSortCostGraph(cg_topo_list_);
  }
  ofs << "op,size\n";
  for (auto &node : cg_topo_list_) {
    size_t size = 0;
    if (!edge_set_.at(node).empty()) {
      size = edge_set_.at(node)[0]->trans_size_;
    }
    ofs << "\"" << node->name_ << "\"," 
        << size << "\n";
  }
  ofs.close();
}

void CostGraph::TopoSortCostGraph(std::vector<NodePtr> &order) {
  if (r_edge_set_.empty()) {
    ConstructReverseEdges();
  }

  std::unordered_map<NodePtr, int> in_degrees;
  for (auto &edges : r_edge_set_) {
    in_degrees[edges.first] = edges.second.size();
  }

  std::queue<NodePtr> q;
  for (auto &p : in_degrees) {
    if (p.second == 0) {
      q.push(p.first);
    }
  }
  while(!q.empty()) {
    auto cur = q.front();
    order.push_back(cur);
    q.pop();
    auto &succ = edge_set_[cur];
    for (auto &end_nodes : succ) {
      if (--in_degrees[end_nodes->end_node_] == 0) {
        q.push(end_nodes->end_node_);
      }
    }    
  }
}

CNodePtrList CostGraph::GenerateGraphSegment(const std::vector<NodePtr> &node_list) {
  CNodePtrList res, tail_load_nodes;
  for (auto &node : node_list) {
    std::transform(node->real_cnodes_.begin(), node->real_cnodes_.end(), std::back_inserter(res),
                   [this, &tail_load_nodes](CNodePtr cnode) {
                     if (fused_cnodes_set.find(AnfAlgo::GetCNodeName(cnode)) != fused_cnodes_set.end()) {
                       auto &load_nodes = fused_node_load_node_map_[cnode];
                       std::copy(load_nodes.begin(), load_nodes.end(), std::back_inserter(tail_load_nodes));
                     }
                     return cnode;
                   });
  }
  // deal with redundant Load nodes
  std::copy(tail_load_nodes.begin(), tail_load_nodes.end(), std::back_inserter(res));
  return res;
}

GraphId CostGraph::GenerateKernelGraphFromSegment(session::SessionPtr &session_impl, KernelGraphPtr &origin_graph, const CNodePtrList &node_list, std::vector<std::string> &output_name) {
  // Prepare: given nodes in execution_order, add other CNodes like TupleGetItem and Load
  // The input CNode list contains all CNodes needed to construct a KernelGraph except the tail MakeTuple, UpdateState, Depend, MakeTuple, and Return
  // Construct a new KernelGraph which is a partial shallow copy of the full KernelGraph
  MS_EXCEPTION_IF_NULL(session_impl);
  MS_EXCEPT_CHECK_NULL(origin_graph);
  if (node_list.empty()) {
    MS_LOG(EXCEPTION) << "Input CNodePtrList is empty";
    return kInvalidGraphId;
  }
  HashMap<AnfNodePtr, AnfNodePtr> old_to_new_map;
  AnfNodePtrList load_nodes;
  CNodePtrList new_exec_order;
  load_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
  auto graph = session_impl->NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_device_target(origin_graph->device_target());
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    // create a new CNode
    CNodePtr new_node;
    AnfNodePtrList new_node_inputs;
    auto prim = AnfAlgo::GetCNodePrimitive(node);
    if (prim != nullptr) {
      new_node_inputs.push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
    } else {
      MS_LOG(EXCEPTION) << "The primitive of " << node->fullname_with_scope() << " is null";
    }
    // construct CNode inputs
    auto& origin_node_inputs = node->inputs();
    for (size_t input_idx = 1; input_idx < origin_node_inputs.size(); ++input_idx) {
      auto& in_node = origin_node_inputs[input_idx];
      MS_EXCEPTION_IF_NULL(in_node);
      if (old_to_new_map.find(in_node) != old_to_new_map.end()) {
        new_node_inputs.push_back(old_to_new_map[in_node]);
        continue;
      } else if (in_node->isa<ValueNode>() && !IsValueNode<FuncGraph>(in_node)) {
        auto value_node = in_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto value = value_node->value();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<None>()) {
          continue;
        }
        auto new_value_node = graph->NewValueNode(value_node);
        graph->AddValueNodeToGraph(new_value_node);
        new_node_inputs.push_back(new_value_node);
        old_to_new_map[in_node] = new_value_node;
      } else if (in_node->isa<Parameter>()) {
        // new parameters
        auto new_param_node = graph->NewParameter(in_node->cast<ParameterPtr>());
        new_param_node->set_kernel_info(in_node->kernel_info_ptr());
        // the name may have some problems
        new_param_node->set_debug_info(in_node->debug_info());
        graph->MutableInputs()->push_back(new_param_node);
        graph->MutableValidInputs()->push_back(true);
        new_node_inputs.push_back(new_param_node);
        old_to_new_map[in_node] = new_param_node;
        continue;
      } else {
        // CNode from other graph (previous), can only be Computation Nodes
        ParameterPtr new_param_node;
        if (!in_node->isa<CNode>()) {
          MS_LOG(EXCEPTION) << "The input of a CNode should be CNode, Parameter, or ValueNode";
        }
        const auto &prim = in_node->cast<CNodePtr>()->inputs().at(0);
        if (!IsOneOfPrimitive(prim, forbidden_cnodes)) {
          // Contains shape and type? (of the output of the origin node)
          new_param_node = graph->NewParameter(in_node->abstract());
          if (IsPrimitive(prim, prim::kPrimTupleGetItem)) {
            auto output_idx = AnfAlgo::GetTupleGetItemOutIndex(in_node->cast<CNodePtr>());
            auto real_kernel = AnfAlgo::VisitKernel(in_node, output_idx);
            auto ref_real_node = real_kernel.first;
            auto ref_real_node_idx = real_kernel.second;
            auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_idx);
            // auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
            auto &k_info_d = ref_real_node->kernel_info_ptr();
            auto k_info = std::dynamic_pointer_cast<device::KernelInfo>(k_info_d);
            MS_EXCEPTION_IF_NULL(k_info);
            // create new KernelInfo
            auto d_kernel_info = std::make_shared<device::KernelInfo>();
            MS_EXCEPTION_IF_NULL(d_kernel_info);
            new_param_node->set_kernel_info(d_kernel_info);
            d_kernel_info->set_graph_id(graph->graph_id());
            d_kernel_info->set_feature_map_flag(k_info->is_feature_map());
            kernel::KernelBuildInfo::KernelBuildInfoBuilder builder(k_info->GetMutableSelectKernelBuildInfo());
            d_kernel_info->set_select_kernel_build_info(builder.Build());
            AnfAlgo::SetOutputAddr(nullptr, 0, new_param_node.get());
            auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), new_param_node->Shape()->cast<abstract::BaseShapePtr>());
            new_param_node->set_abstract(abstract);
            // rename
            auto ref_real_node_cg = TranslateAnfNode(origin_graph, graph_, ref_real_node);
            auto cg_node_list_it = real_fake_map_.find(ref_real_node_cg);
            if (cg_node_list_it == real_fake_map_.end()) {
              MS_LOG(EXCEPTION) << "Cannot find CostGraph::Node for CNode named " << ref_real_node->fullname_with_scope();
            }
            new_param_node->set_debug_info(std::make_shared<NodeDebugInfo>(cg_node_list_it->second.back()->name_));
            new_param_node->set_name(new_param_node->DebugString());
          } else {
            auto &k_info_d = in_node->kernel_info_ptr();
            auto k_info = std::dynamic_pointer_cast<device::KernelInfo>(k_info_d);
            MS_EXCEPTION_IF_NULL(k_info);
            // create new KernelInfo
            auto d_kernel_info = std::make_shared<device::KernelInfo>();
            MS_EXCEPTION_IF_NULL(d_kernel_info);
            new_param_node->set_kernel_info(d_kernel_info);
            d_kernel_info->set_graph_id(graph->graph_id());
            d_kernel_info->set_feature_map_flag(k_info->is_feature_map());
            kernel::KernelBuildInfo::KernelBuildInfoBuilder builder(k_info->GetMutableSelectKernelBuildInfo());
            d_kernel_info->set_select_kernel_build_info(builder.Build());
            AnfAlgo::SetOutputAddr(nullptr, 0, new_param_node.get());
            // rename
            auto in_node_cg = TranslateAnfNode(origin_graph, graph_, in_node);
            auto cg_node_list_it = real_fake_map_.find(in_node_cg);
            if (cg_node_list_it == real_fake_map_.end()) {
              MS_LOG(EXCEPTION) << "Cannot find CostGraph::Node for CNode named " << in_node->fullname_with_scope();
            }
            new_param_node->set_debug_info(std::make_shared<NodeDebugInfo>(cg_node_list_it->second.back()->name_));
            new_param_node->set_name(new_param_node->DebugString());
          }
        } else {
          MS_LOG(EXCEPTION) << "Unsupport CNode as Parameter, check the input CNode seq";
          return kInvalidGraphId;
        }
        // add to input list
        new_param_node->IncreaseUsedGraphCount();
        graph->MutableInputs()->push_back(new_param_node);
        graph->MutableValidInputs()->push_back(true);
        // Input parameter does not need Load
        new_node_inputs.push_back(new_param_node);
        old_to_new_map[in_node] = new_param_node;
      }
    }
    new_node = graph->NewCNode(new_node_inputs);
    MS_EXCEPTION_IF_NULL(new_node);
    new_node->CloneCNodeInfo(node);

    if (IsPrimitiveCNode(new_node, prim::kPrimLoad)) {
      auto &load_user_map = origin_graph->manager()->node_users();
      auto it = load_user_map.find(node);
      if (it != load_user_map.end()) {
        load_nodes.push_back(new_node);
      } else {
        MS_LOG(EXCEPTION) << "Load node for origin graph " << node->fullname_with_scope() << " has no user nodes";
        return kInvalidGraphId;
      }
    } else {
      auto &k_info_d = node->kernel_info_ptr();
      auto k_info = std::dynamic_pointer_cast<device::KernelInfo>(k_info_d);
      MS_EXCEPTION_IF_NULL(k_info);
      auto d_kernel_info = std::make_shared<device::KernelInfo>();
      MS_EXCEPTION_IF_NULL(d_kernel_info);
      new_node->set_kernel_info(d_kernel_info);
      d_kernel_info->set_graph_id(graph->graph_id());
      d_kernel_info->set_feature_map_flag(k_info->is_feature_map());
      d_kernel_info->set_kernel_mod(k_info->GetKernelModPtr());
      if (AnfUtils::IsRealKernel(new_node)) {
        kernel::KernelBuildInfo::KernelBuildInfoBuilder builder(k_info->GetMutableSelectKernelBuildInfo());
        d_kernel_info->set_select_kernel_build_info(builder.Build());
      }
      AnfAlgo::SetOutputAddr(nullptr, 0, new_node.get());
    }

    old_to_new_map[node] = new_node;
    if (AnfUtils::IsRealKernel(new_node) && !opt::IsNopNode(new_node)) {
      new_exec_order.push_back(new_node);
    }
  }
  // Construct tail nodes
  auto make_tup_load = graph->NewCNode(load_nodes);
  auto u = NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto update_state_cnode = graph->NewCNode({NewValueNode(prim::kPrimUpdateState), u, make_tup_load});
  update_state_cnode->set_abstract(kUMonad->ToAbstract());
  // construct output
  AnfNodePtrList output_nodes;
  GetSegmentOutput(origin_graph, node_list, output_nodes);
  if (output_nodes.empty()) {
    MS_LOG(EXCEPTION) << "Cannot generate output from the segment";
    return kInvalidGraphId;
  }
  AnfNodePtr real_output;
  if (output_nodes.size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(output_nodes), std::end(output_nodes), std::back_inserter(output_args),
                         [this, &origin_graph, &old_to_new_map, &output_name](const AnfNodePtr &node) -> AnfNodePtr { 
                          auto node_cg = TranslateAnfNode(origin_graph, this->graph_, node);
                          auto cg_node_list_it = real_fake_map_.find(node_cg);
                          if (cg_node_list_it == real_fake_map_.end()) {
                            MS_LOG(EXCEPTION) << "Cannot find CostGraph::Node for CNode named " << node->fullname_with_scope();
                          }
                          output_name.push_back(cg_node_list_it->second.back()->name_);
                          return old_to_new_map[node]; 
                        });
    real_output = graph->NewCNode(output_args);
  } else {
    real_output = old_to_new_map[output_nodes[0]];
    auto output_node_0_cg = TranslateAnfNode(origin_graph, graph_, output_nodes[0]);
    auto cg_node_list_it = real_fake_map_.find(output_node_0_cg);
    if (cg_node_list_it == real_fake_map_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find CostGraph::Node for CNode named " << output_nodes[0]->fullname_with_scope();
    }
    output_name.push_back(cg_node_list_it->second.back()->name_);
  }
  auto depend_node = graph->NewCNode({NewValueNode(prim::kPrimDepend), real_output, update_state_cnode});
  auto return_node = graph->NewCNode({NewValueNode(prim::kPrimReturn), depend_node});
  graph->set_return(return_node);
  graph->set_output(depend_node);
  // manage generated graph
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->set_execution_order(new_exec_order);
  graph->MutableInputNodes()->clear();
  std::copy(graph->inputs().begin(), graph->inputs().end(), std::back_inserter(*graph->MutableInputNodes()));
  session_impl->SetRuntimeResourceForFakeGraph(graph);
  return graph->graph_id();
}

std::unordered_map<std::string, float>& CostGraph::GetCloudTimeMap(size_t bsz) {
  auto it = cloud_time_map_.find(bsz);
  if (it == cloud_time_map_.end()) {
    MS_LOG(EXCEPTION) << "CostGraph::GetCloudTimeMap: cannot find cloud time map with batch size = " << bsz;
  }
  return it->second;
}

void CostGraph::SetCloudTimeMap(GraphProfile &remote_time_map, size_t default_bsz) {
  // we assume all profile share the same key set
  for (const auto& batch_entry : remote_time_map.entries()) {
    cloud_time_map_[batch_entry.batch_size()] = std::unordered_map<std::string, float>();
    auto &time_map = cloud_time_map_.at(batch_entry.batch_size());
    for (const auto& profile_entry : batch_entry.profile()) {
      time_map[profile_entry.name()] = profile_entry.time();
    }
  }

  auto time_map_1_it = cloud_time_map_.find(default_bsz);
  if (time_map_1_it == cloud_time_map_.end()) {
    MS_LOG(EXCEPTION) << "CostGraph::SetCloudTimeMap: time map with batch size = 1 not found";
  }
  auto &time_map = time_map_1_it->second;
  for (auto &n : node_set_) {
    auto time_it = time_map.find(n->name_);
    if (time_it == time_map.end()) {
      if (n != source_node_ && n != sink_node_) {
        MS_LOG(EXCEPTION) << "CostGraph::SetCloudTimeMap: cannot find profile result for node " << n->name_;
      } else {
        n->remote_exec_time_ = 0.0;
      }
      continue;
    }
    n->remote_exec_time_ = time_it->second;
  }
}

std::vector<size_t> CostGraph::GetOutputShape(const NodePtr &node) {
  NodePtr real_node = node;
  while (real_node->real_cnodes_.empty() && !real_node->inputs_.empty()) {
    real_node = real_node->inputs_.back();
  }
  if (real_node->real_cnodes_.empty()) {
    return std::vector<size_t>();
  }
  CNodePtr &real_cnode = real_node->real_cnodes_.back();
  MS_EXCEPTION_IF_NULL(real_cnode);
  return AnfAlgo::GetOutputDeviceShape(real_cnode, 0);
}

double CostGraph::GetFullLocalTime() {
  return std::accumulate(node_set_.begin(), node_set_.end(), 0.0, [](double partial_sum, const NodePtr &node) {
    return partial_sum + node->local_exec_time_;
  });
}

void LatencyGraph::PathIdxTrie::Insert(const std::vector<int> &pidx) {
  TrieNodePtr cur_node = root_;
  for (auto i : pidx) {
    auto &children = cur_node->children_;
    auto cur_len = (int)children.size();
    if (i >= cur_len) {
      auto insert_size = i - cur_len + 1;
      children.insert(children.end(), insert_size, nullptr);
    }
    if (children[i] == nullptr) {
      children[i] = std::make_shared<TrieNode>();
      children[i]->parent_ = cur_node.get();
    }
    cur_node = children[i];
  }
  cur_node->is_end_ = true;
}

void LatencyGraph::ConstructLatencyGraph(size_t start_idx, size_t end_idx, std::shared_ptr<CostGraph> &cost_graph) {
  auto &topo_list = cost_graph->cg_topo_list_;
  auto &cost_graph_edge_set = cost_graph->edge_set_;

  start_node_ = topo_list[start_idx];
  end_node_ = topo_list[end_idx];
  if (cost_graph_edge_set[start_node_].size() == 1 &&
      cost_graph_edge_set[start_node_][0]->end_node_ == end_node_) {
    is_chain_ = true;
  } else {
    is_chain_ = false;
  }

  for (size_t i = start_idx; i <= end_idx; ++i) {
    edge_set_[topo_list[i]] = std::vector<EdgeNodePtr>();
    r_edge_set_[topo_list[i]] = std::vector<EdgeNodePtr>();
  }

  for (size_t i = start_idx; i <= end_idx; ++i) {
    node_set_.emplace(topo_list[i]);
    if (i != end_idx) {
      for (auto &e : cost_graph_edge_set[topo_list[i]]) {
        edge_set_[topo_list[i]].emplace_back(std::make_shared<EdgeNode>(e));
      }
    }
  }

  for (auto &es : edge_set_) {
    for (size_t i = 0; i < es.second.size(); ++i) {
      auto &e = es.second[i];
      auto r_e = std::make_shared<EdgeNode>(es.first, e->trans_size_);
      r_edge_set_[e->end_node_].emplace_back(r_e);
      r_e->SetReverseEdge(e);
      e->SetReverseEdge(r_e);
    }
  }
}

void LatencyGraph::MarkLevel() {
  std::queue<CostGraph::NodePtr> q;
  std::unordered_map<CostGraph::NodePtr, int> in_degrees;
  for (auto &edges : r_edge_set_) {
    in_degrees[edges.first] = edges.second.size();
  }

  for (auto &p : in_degrees) {
    if (p.second == 0) {
      q.push(p.first);
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL((q.size() == 1), "There should be one and only one node with in degree = 0");

  while (!q.empty()) {
    CostGraph::NodePtr cur_node = q.front();
    local_topo_list_.emplace_back(cur_node);
    q.pop();

    auto &r_in_edges = r_edge_set_.at(cur_node);
    auto &out_edges = edge_set_.at(cur_node);
    auto in_degree = r_in_edges.size();
    auto out_degree = out_edges.size();
    if (cur_node == start_node_) {
      for (auto &e : out_edges) {
        e->level_.push_back(1);
      }
    } else if (cur_node != end_node_) {
      // visiting by topo order ensures that when visiting a node, all its parent nodes are visited before
      if (in_degree == 1 && out_degree == 1) {
        out_edges[0]->level_ = r_in_edges[0]->r_edge_->level_;
      } else if (in_degree == 1 && out_degree > 1) {
        for (auto &e : out_edges) {
          e->level_ = r_in_edges[0]->r_edge_->level_;
          int l = GetLevel(e) + 1;
          e->level_.push_back(l);
        }
      } else if (in_degree > 1 && out_degree == 1) {
        int min_level = INT_MAX;
        std::vector<int> *lp = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e->r_edge_);
          if (l < min_level) {
            min_level = l;
            lp = &r_e->r_edge_->level_;
          }
        }
        out_edges[0]->level_ = *lp;
        if (!lp->empty()) {
          out_edges[0]->level_.pop_back();
        }
      } else {
        int min_level = INT_MAX;
        std::vector<int> *lp = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e->r_edge_);
          if (l < min_level) {
            min_level = l;
            lp = &r_e->r_edge_->level_;
          }
        }
        for (auto &e : out_edges) {
          e->level_ = *lp;
        }
      }
    }

    for (const auto &end_nodes : out_edges) {
      if (--in_degrees[end_nodes->end_node_] == 0) {
        q.push(end_nodes->end_node_);
      }
    }
  }
}

void LatencyGraph::ReviseLevel() {
  std::queue<CostGraph::NodePtr> q;
  std::unordered_map<CostGraph::NodePtr, int> in_degrees;
  // Notice: reverse graph
  for (auto &edges : edge_set_) {
    in_degrees[edges.first] = edges.second.size();
  }

  for (auto &p : in_degrees) {
    if (p.second == 0) {
      q.push(p.first);
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL((q.size() == 1), "There should be one and only one node with out degree = 0");

  while (!q.empty()) {
    CostGraph::NodePtr cur_node = q.front();
    q.pop();

    // Notice: reverse graph
    auto &r_in_edges = edge_set_.at(cur_node);
    auto &out_edges = r_edge_set_.at(cur_node);
    auto in_degree = r_in_edges.size();
    auto out_degree = out_edges.size();
    if (cur_node == end_node_) {
      for (auto &e : out_edges) {
        e->level_.push_back(1);
        UpdateLevelVector(e);
      }
    } else if (cur_node != start_node_) {
      // visiting by topo order ensures that when visiting a node, all its parent nodes are visited before
      if (in_degree == 1 && out_degree == 1) {
        out_edges[0]->level_ = r_in_edges[0]->r_edge_->level_;
        UpdateLevelVector(out_edges[0]);
      } else if (in_degree == 1 && out_degree > 1) {
        for (auto &e : out_edges) {
          e->level_ = r_in_edges[0]->r_edge_->level_;
          int l = GetLevel(e) + 1;
          e->level_.push_back(l);
          UpdateLevelVector(e);
        }
      } else if (in_degree > 1 && out_degree == 1) {
        int min_level = INT_MAX;
        std::vector<int> *lp = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e->r_edge_);
          if (l < min_level) {
            min_level = l;
            lp = &r_e->r_edge_->level_;
          }
        }
        out_edges[0]->level_ = *lp;
        if (!lp->empty()) {
          out_edges[0]->level_.pop_back();
        }
        UpdateLevelVector(out_edges[0]);
      } else {
        int min_level = INT_MAX;
        std::vector<int> *lp = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e->r_edge_);
          if (l < min_level) {
            min_level = l;
            lp = &r_e->r_edge_->level_;
          }
        }
        for (auto &e : out_edges) {
          e->level_ = *lp;
          UpdateLevelVector(e);
        }
      }
    }

    for (const auto &end_nodes : out_edges) {
      if (--in_degrees[end_nodes->end_node_] == 0) {
        q.push(end_nodes->end_node_);
      }
    }
  }
}

void LatencyGraph::IdentifyPaths() {
  std::map<std::vector<int>, std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>>> tmp_path_set;
  std::queue<CostGraph::NodePtr> q;
  std::unordered_map<CostGraph::NodePtr, int> in_degrees;
  for (auto &edges : r_edge_set_) {
    in_degrees[edges.first] = edges.second.size();
  }

  for (auto &p : in_degrees) {
    if (p.second == 0) {
      q.push(p.first);
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL((q.size() == 1), "There should be one and only one node with in degree = 0");

  auto level_cmp_func = [this](EdgeNodePtr &lhs, EdgeNodePtr &rhs) {
    return (GetLevel(lhs->r_edge_) < GetLevel(rhs->r_edge_)) || (GetLevel(lhs->r_edge_) == GetLevel(rhs->r_edge_) && lhs->end_node_->name_ < rhs->end_node_->name_);
  };

  while (!q.empty()) {
    CostGraph::NodePtr cur_node = q.front();
    q.pop();

    auto &r_in_edges = r_edge_set_.at(cur_node);
    auto &out_edges = edge_set_.at(cur_node);
    auto in_degree = r_in_edges.size();
    auto out_degree = out_edges.size();
    if (cur_node == start_node_) {
      std::sort(out_edges.begin(), out_edges.end(), level_cmp_func);
      std::vector<int> cur_pidx = std::vector<int>();
      MS_EXCEPTION_IF_CHECK_FAIL((GetLevel(out_edges[0]->r_edge_) == 1), "Min out edge level should be 1 for the start node");
      MarkPathByOrder(out_edges, 0, cur_pidx);
    } else if (cur_node != end_node_) {
      // visiting by topo order ensures that when visiting a node, all its parent nodes are visited before
      if (in_degree == 1 && out_degree == 1) {
        out_edges[0]->pidx_ = r_in_edges[0]->r_edge_->pidx_;
      } else if (in_degree == 1 && out_degree > 1) {
        std::sort(out_edges.begin(), out_edges.end(), level_cmp_func);
        std::vector<int> cur_pidx = r_in_edges[0]->r_edge_->pidx_;
        MS_EXCEPTION_IF_CHECK_FAIL((GetLevel(out_edges[0]->r_edge_) == (int)cur_pidx.size() + 1), "Min out edge level should be len(base pidx) + 1");
        MarkPathByOrder(out_edges, GetLevel(r_in_edges[0]), cur_pidx);
      } else if (in_degree > 1 && out_degree == 1) {
        int min_level = INT_MAX;
        std::vector<int> *ip = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e);
          if (l < min_level) {
            min_level = l;
            ip = &r_e->r_edge_->pidx_;
          }
        }
        out_edges[0]->pidx_ = *ip;
        if (min_level > 0) {
          MS_EXCEPTION_IF_CHECK_FAIL((GetLevel(out_edges[0]->r_edge_) == min_level - 1), "In COV, the obtained pidx len should agree with the level");
          out_edges[0]->pidx_.resize(min_level - 1);
        }
      } else {
        int min_level = INT_MAX;
        std::vector<int> *ip = nullptr;
        for (auto &r_e : r_in_edges) {
          int l = GetLevel(r_e);
          if (l < min_level) {
            min_level = l;
            ip = &r_e->r_edge_->pidx_;
          }
        }

        std::sort(out_edges.begin(), out_edges.end(), level_cmp_func);
        std::vector<int> cur_pidx = *ip;
        if (min_level > 0) {
          cur_pidx.resize(min_level - 1);
        }
        MS_EXCEPTION_IF_CHECK_FAIL((GetLevel(out_edges[0]->r_edge_) == (int)cur_pidx.size() + 1), "Min out edge level should be len(base pidx) + 1");
        MarkPathByOrder(out_edges, min_level - 1, cur_pidx);
      }
    }

    for (const auto &end_nodes : out_edges) {
      tmp_path_set[end_nodes->pidx_].emplace_back(std::make_pair(cur_node, end_nodes));
      if (--in_degrees[end_nodes->end_node_] == 0) {
        q.push(end_nodes->end_node_);
      }
    }
  }
  ConstructSimpleGraphs(tmp_path_set);
}

void LatencyGraph::ApplyCpsRatio(std::pair<double, double> &qnt_estimator, std::pair<double, double> &source_prof, std::pair<double, double> &avg_prof, std::unordered_set<std::string> &enable_cps_nodes) {
  ApplyCpsRatioInner(trie_.root_, qnt_estimator, source_prof, avg_prof, enable_cps_nodes);
}

double LatencyGraph::GetFullLocalTimeWithoutST() {
  double res = 0.0;
  for (auto &n : node_set_) {
    if (n != start_node_ && n != end_node_) {
      res += n->local_exec_time_;
    }
  }
  return res;
}

double LatencyGraph::GetFullRemoteTimeWithoutST() {
  double res = 0.0;
  for (auto &n : node_set_) {
    if (n != start_node_ && n != end_node_) {
      res += n->remote_exec_time_;
    }
  }
  return res;
}

void LatencyGraph::GetTopoSortFromCutPoints(const std::unordered_set<CostGraph::NodePtr> &cut_nodes, std::vector<CostGraph::NodePtr> &res) {
  std::queue<CostGraph::NodePtr> q;
  std::unordered_map<CostGraph::NodePtr, bool> vis;
  std::unordered_set<CostGraph::NodePtr> res_set;
  for (auto &n : node_set_) {
    vis[n] = false;
    if (cut_nodes.find(n) != cut_nodes.end()) {
      q.emplace(n);
    }
  }

  while (!q.empty()) {
    CostGraph::NodePtr cur_node = q.front();
    vis[cur_node] = true;
    res_set.emplace(cur_node);
    q.pop();

    auto &out_edges = r_edge_set_.at(cur_node);
    for (const auto &end_nodes : out_edges) {
      if (!vis[end_nodes->end_node_]) {
        q.push(end_nodes->end_node_);
      }
    }
  }

  for (auto &n : local_topo_list_) {
    if (res_set.find(n) == res_set.end()) {
      res.emplace_back(n);
    }
  }
}

void LatencyGraph::GetTopoSortFromCutPointsReverse(const std::unordered_set<CostGraph::NodePtr> &cut_nodes, std::vector<CostGraph::NodePtr> &res) {
  std::queue<CostGraph::NodePtr> q;
  std::unordered_map<CostGraph::NodePtr, bool> vis;
  std::unordered_set<CostGraph::NodePtr> res_set;
  for (auto &n : node_set_) {
    vis[n] = false;
    if (cut_nodes.find(n) != cut_nodes.end()) {
      q.emplace(n);
    }
  }

  while (!q.empty()) {
    CostGraph::NodePtr cur_node = q.front();
    vis[cur_node] = true;
    res_set.emplace(cur_node);
    q.pop();

    auto &out_edges = r_edge_set_.at(cur_node);
    for (const auto &end_nodes : out_edges) {
      if (!vis[end_nodes->end_node_]) {
        q.push(end_nodes->end_node_);
      }
    }
  }

  for (auto &n : local_topo_list_) {
    if (res_set.find(n) != res_set.end()) {
      res.emplace_back(n);
    }
  }
}

void LatencyGraph::DrawLatencyGraph(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }
  ofs << "digraph g {\n";
  ofs << "  fontname = \"Courier New\"\n";
  ofs << "  node [ fontname = \"Courier New\" ]\n";
  ofs << "  edge [ fontname = \"Courier New\" ]\n";
  ofs << "  graph [ fontsize = 24, spline = true, overlap = false, compound = true];\n";
  ofs << "  ratio = auto;\n";
  for (auto &node : node_set_) {
    MS_EXCEPTION_IF_NULL(node);
    ofs << "  \"" << node->name_ << "\" [ label = \"" << node->name_ << "\", color = dimgray ];\n";
  }

  for (auto &node : node_set_) {
    auto &edge_list = edge_set_[node];
    for (auto &edge : edge_list) {
      MS_EXCEPTION_IF_NULL(edge);
      ofs << "  \"" << node->name_ << "\" -> \"" << edge->end_node_->name_ << "\" [ label = <<font color=\"crimson\">";
      PrintVector(edge->r_edge_->level_, ofs);
      ofs << "; </font><font color=\"cornflowerblue\">";
      PrintVector(edge->pidx_, ofs);
      ofs << "</font>>, color = dimgray ];\n";
    }
  }
  ofs << "}";
  ofs.close();
}

void LatencyGraph::DrawLatencyGraphAsSubgraph(size_t idx, std::ofstream &ofs) {
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "ofstream is invalid";
    return;
  }
  ofs << "  subgraph \"cluster_" << idx << "\" {\n";
  for (auto &node : node_set_) {
    MS_EXCEPTION_IF_NULL(node);
    ofs << "    \"" << node->name_ << idx << "\" [ label = \"" << node->name_ << "\", color = dimgray ];\n";
  }

  for (auto &node : node_set_) {
    auto &edge_list = edge_set_[node];
    for (auto &edge : edge_list) {
      ofs << "    \"" << node->name_ << idx << "\" -> \"" << edge->end_node_->name_ << idx << "\" [ label = <<font color=\"crimson\">";
      PrintVector(edge->r_edge_->level_, ofs);
      ofs << "; </font><font color=\"cornflowerblue\">";
      PrintVector(edge->pidx_, ofs);
      ofs << "</font>>, color = dimgray ];\n";
    }
  }
  ofs << "  }\n\n";
}

int LatencyGraph::GetLevel(EdgeNodePtr &e) {
  MS_EXCEPTION_IF_NULL(e);
  return (e->level_.empty()) ? 0 : e->level_.back();
}

int LatencyGraph::GetLevel(EdgeNode *e) {
  MS_EXCEPTION_IF_NULL(e);
  return (e->level_.empty()) ? 0 : e->level_.back();
}

void LatencyGraph::UpdateLevelVector(EdgeNodePtr &e) {
  MS_EXCEPTION_IF_NULL(e);
  if (e->level_.size() < e->r_edge_->level_.size()) {
    e->level_ = e->r_edge_->level_;
  }
}

void LatencyGraph::CheckIsChain(std::shared_ptr<LatencyGraph> &g) {
  MS_EXCEPTION_IF_NULL(g);
  g->is_chain_ = true;
  for (auto &p : g->edge_set_) {
    if (p.first != g->end_node_) g->is_chain_ &= (p.second.size() == 1);
  }
}

void LatencyGraph::ConstructTimeArrays(std::shared_ptr<LatencyGraph> &g) {
  MS_EXCEPTION_IF_NULL(g);
  if (!g->is_chain_) {
    MS_LOG(EXCEPTION) << "LatencyGraph: cannot construct prefix and suffix sum arrays for non-chain simple LatencyGraph";
  }

  g->local_topo_list_.clear();
  CostGraph::NodePtr cur_node = g->start_node_;
  while (true) {
    g->local_topo_list_.emplace_back(cur_node);
    if (cur_node != g->end_node_) {
      if (g->edge_set_.at(cur_node)[0]->subgraphs_.empty()) {
        double trans_size = (double)g->edge_set_.at(cur_node)[0]->trans_size_;
        g->size_list_.push_back(trans_size); // in B
      } else {
        g->size_list_.push_back(-1.0);
        g->idx_to_complex_edges_[g->local_topo_list_.size() - 1] = g->edge_set_.at(cur_node)[0];
      }
      cur_node = g->edge_set_.at(cur_node)[0]->end_node_;
    } else {
      break;
    }
  }
  
  size_t num_nodes = g->local_topo_list_.size();
  g->local_time_pre_sum_.resize(num_nodes + 1, 0.0);
  g->remote_time_suf_sum_.resize(num_nodes + 1, 0.0);
  for (size_t i = 1; i <= num_nodes; ++i) {
    g->local_time_pre_sum_[i] = g->local_time_pre_sum_[i - 1] + g->local_topo_list_[i - 1]->local_exec_time_ / 1e3; // in ms
    g->remote_time_suf_sum_[num_nodes - i] = g->remote_time_suf_sum_[num_nodes - i + 1] + g->local_topo_list_[num_nodes - i]->remote_exec_time_ / 1e3; // in ms
  }
}

void LatencyGraph::MarkPathByOrder(std::vector<EdgeNodePtr> &out_edges, int last_level, std::vector<int> cur_pidx) {
  bool first_enter_flag = true;
  for (auto &e : out_edges) {
    int cur_level = GetLevel(e->r_edge_);
    if (!first_enter_flag) {
      cur_pidx.back()++;
    } else {
      first_enter_flag = false;
    }
    if (cur_level != last_level) {
      cur_pidx.push_back(0);
    }
    e->pidx_ = cur_pidx;
    last_level = cur_level;
  }
}

void LatencyGraph::ConstructSimpleGraphs(std::map<std::vector<int>, std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>>> &tmp_path_set) {
  // construct a prefix tree on pidx to assist further construction
  for (auto &p : tmp_path_set) {
    trie_.Insert(p.first);
  }
  std::vector<int> cur_pidx;
  ConstructSimpleGraphsInner(trie_.root_, cur_pidx, tmp_path_set);
}

void LatencyGraph::ConstructSimpleGraphsInner(TrieNodePtr root, std::vector<int> &cur_pidx, const std::map<std::vector<int>, std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>>> &tmp_path_set) {
  for (size_t i = 0; i < root->children_.size(); ++i) {
    if (root->children_[i]) {
      cur_pidx.push_back(i);
      ConstructSimpleGraphsInner(root->children_[i], cur_pidx, tmp_path_set);
      cur_pidx.pop_back();
    }
  }
  if (root->is_end_) {
    auto &cur_pidx_edges = tmp_path_set.at(cur_pidx);
    std::vector<std::shared_ptr<LatencyGraph>> children_graphs;
    // collect all children graphs
    for (auto &c : root->children_) {
      if (c && c->is_end_) {
        for (auto &g : c->simple_graphs_) {
          children_graphs.emplace_back(g);
        }

        for (auto &n : c->exempt_sta_nodes_) {
          root->exempt_sta_nodes_.insert(n);
        }
      }
    }
    std::vector<std::shared_ptr<LatencyGraph>> cur_node_graphs;
    ConstructUpperLevelSimpleGraph(root, cur_pidx, cur_pidx_edges, children_graphs, cur_node_graphs, root->exempt_sta_nodes_);
    std::transform(cur_node_graphs.begin(), cur_node_graphs.end(), std::back_inserter(root->simple_graphs_), [this](std::shared_ptr<LatencyGraph> &g) {
      ConstructTimeArrays(g);
      return g;
    });
    // keep every children simple graphs paired
    std::sort(root->simple_graphs_.begin(), root->simple_graphs_.end(), [this](std::shared_ptr<LatencyGraph> &lhs, std::shared_ptr<LatencyGraph> &rhs) {
      return lhs->start_node_->name_ < rhs->start_node_->name_;
    });
  }
}

void LatencyGraph::ConstructUpperLevelSimpleGraph(TrieNodePtr &root, const std::vector<int> &cur_pidx, const std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>> &edges, std::vector<std::shared_ptr<LatencyGraph>> &children_graphs, std::vector<std::shared_ptr<LatencyGraph>> &cur_node_graphs, std::unordered_set<CostGraph::NodePtr> &exempt_sta) {
  auto subgraph = std::make_shared<LatencyGraph>();
  auto &sub_node_set = subgraph->node_set_;
  auto &sub_edge_set = subgraph->edge_set_;
  auto &sub_r_edge_set = subgraph->r_edge_set_;
  
  for (auto &p : edges) {
    sub_node_set.insert(p.first);
    sub_edge_set[p.first].emplace_back(p.second);
    auto r_edge_it = std::find_if(r_edge_set_[p.second->end_node_].begin(), r_edge_set_[p.second->end_node_].end(), [&p](EdgeNodePtr &e) {
      return (e->end_node_ == p.first);
    });
    sub_r_edge_set[p.second->end_node_].emplace_back(*r_edge_it);
  }
  // create edges for next_level_graphs
  std::vector<int> cur_level;
  for (size_t i = 1; i <= cur_pidx.size(); ++i) cur_level.push_back(i);
  for (auto &g : children_graphs) {
    MS_EXCEPTION_IF_NULL(g);
    // search for existing start&end nodes
    sub_node_set.insert(g->start_node_);
    sub_node_set.insert(g->end_node_);

    EdgeNodePtr s_e = nullptr, e_s = nullptr;
    auto s_it = sub_edge_set.find(g->start_node_);
    if (s_it == sub_edge_set.end()) {
      // start_node not added in edge_set
      s_e = std::make_shared<EdgeNode>(g->end_node_, 0);
      s_e->level_ = cur_level;
      s_e->pidx_ = cur_pidx;
      sub_edge_set[g->start_node_].emplace_back(s_e);
      s_e->subgraphs_.emplace_back(g);
      root->complex_edge_part_res_[s_e] = PartitionResult();
    } else {
      // find s->e edge
      auto s_e_it = std::find_if(s_it->second.begin(), s_it->second.end(), [&g](EdgeNodePtr &e) {
        return (e->end_node_ == g->end_node_);
      });
      if (s_e_it == s_it->second.end()) {
        // may never be taken?
        s_e = std::make_shared<EdgeNode>(g->end_node_, 0);
        s_e->level_ = cur_level;
        s_e->pidx_ = cur_pidx;
        sub_edge_set[g->start_node_].emplace_back(s_e);
        s_e->subgraphs_.emplace_back(g);
        root->complex_edge_part_res_[s_e] = PartitionResult();
      } else {
        // add graph to s->e edge
        s_e = *s_e_it;
        s_e->subgraphs_.emplace_back(g);
      }
    }

    auto e_it = sub_r_edge_set.find(g->end_node_);
    if (e_it == sub_r_edge_set.end()) {
      // start_node not added in edge_set
      e_s = std::make_shared<EdgeNode>(g->start_node_, 0);
      sub_r_edge_set[g->end_node_].emplace_back(e_s);
    } else {
      // find s->e edge
      auto e_s_it = std::find_if(e_it->second.begin(), e_it->second.end(), [&g](EdgeNodePtr &e) {
        return (e->end_node_ == g->start_node_);
      });
      if (e_s_it == e_it->second.end()) {
        // may never be taken?
        e_s = std::make_shared<EdgeNode>(g->start_node_, 0);
        sub_r_edge_set[g->end_node_].emplace_back(e_s);
      } else {
        // add graph to s->e edge
        e_s = *e_s_it;
      }
    }
    s_e->SetReverseEdge(e_s);
    e_s->SetReverseEdge(s_e);
  }
  // find start & end nodes, may find more than one start/end nodes -> sort according to topo order
  std::unordered_set<CostGraph::NodePtr> tmp_start_nodes, tmp_end_nodes;
  for (auto &p : sub_edge_set) {
    if (sub_r_edge_set.find(p.first) == sub_r_edge_set.end()) {
      tmp_start_nodes.insert(p.first);
    }
  }
  for (auto &p : sub_r_edge_set) {
    if (sub_edge_set.find(p.first) == sub_edge_set.end()) {
      tmp_end_nodes.insert(p.first);
    }
  }

  for (auto &n : tmp_start_nodes) {
    sub_r_edge_set[n] = std::vector<EdgeNodePtr>();
  }
  for (auto &n : tmp_end_nodes) {
    sub_node_set.insert(n);
    sub_edge_set[n] = std::vector<EdgeNodePtr>();
  }
  // find all STA nodes
  std::unordered_set<CostGraph::NodePtr> sta_nodes;
  for (auto &n : sub_node_set) {
    int in_degree = r_edge_set_[n].size();
    int out_degree = edge_set_[n].size();
    if (in_degree > 1 && out_degree > 1 && exempt_sta.find(n) == exempt_sta.end()) {
      sta_nodes.insert(n);
      exempt_sta.insert(n);
    }
  }
  // if no separated grahs and STA nodes, set start/end nodes & check is_chain
  if ((tmp_start_nodes.size() == 1) 
      && (tmp_end_nodes.size() == 1) 
      && (sta_nodes.empty())) {
    subgraph->start_node_ = *tmp_start_nodes.begin();
    subgraph->end_node_ = *tmp_end_nodes.begin();
    CheckIsChain(subgraph);
    cur_node_graphs.emplace_back(subgraph);
  } else {
    // graph need to be split further if it contains STA nodes / separated graphs
    SplitSimpleGraph(subgraph, cur_node_graphs, tmp_start_nodes, tmp_end_nodes, sta_nodes);
  }
}

void LatencyGraph::SplitSimpleGraph(std::shared_ptr<LatencyGraph> &origin_graph, std::vector<std::shared_ptr<LatencyGraph>> &split_graphs, 
                      std::unordered_set<CostGraph::NodePtr> &tmp_start_nodes, std::unordered_set<CostGraph::NodePtr> &tmp_end_nodes, std::unordered_set<CostGraph::NodePtr> &sta_nodes) {
  // sort STA, start/end nodes according to the topo order of the full graph
  MS_EXCEPTION_IF_CHECK_FAIL((tmp_start_nodes.size() == tmp_end_nodes.size()), "Start nodes and End nodes should be paired");
  std::vector<CostGraph::NodePtr> sorted_start_nodes, sorted_end_nodes;
  for (auto &n : local_topo_list_) {
    auto start_it = tmp_start_nodes.find(n);
    if (start_it != tmp_start_nodes.end()) {
      sorted_start_nodes.emplace_back(*start_it);
    }
    auto end_it = tmp_end_nodes.find(n);
    if (end_it != tmp_end_nodes.end()) {
      sorted_end_nodes.emplace_back(*end_it);
    }
  }
  // split separate graphs
  for (size_t i = 0; i < sorted_start_nodes.size(); ++i) {
    auto new_graph = ConstructGraphFromGraph(origin_graph, sorted_start_nodes[i], sorted_end_nodes[i]);
    split_graphs.emplace_back(new_graph);
  }
  // split by sta nodes
  for (auto &sta_n : sta_nodes) {
    auto g_it = std::find_if(split_graphs.begin(), split_graphs.end(), [&sta_n](std::shared_ptr<LatencyGraph> &g) {
      return (g->node_set_.find(sta_n) != g->node_set_.end());
    });
    auto new_graph_a = ConstructGraphFromGraph(*g_it, (*g_it)->start_node_, sta_n);
    auto new_graph_b = ConstructGraphFromGraph(*g_it, sta_n, (*g_it)->end_node_);
    g_it = split_graphs.erase(g_it);
    g_it = split_graphs.emplace(g_it, new_graph_a);
    split_graphs.emplace(g_it, new_graph_b);
  }
}

std::shared_ptr<LatencyGraph> LatencyGraph::ConstructGraphFromGraph(const std::shared_ptr<LatencyGraph> &origin_graph, const CostGraph::NodePtr &start_node, const CostGraph::NodePtr &end_node) {
  auto new_graph = std::make_shared<LatencyGraph>();
  auto start_it = std::find(local_topo_list_.begin(), local_topo_list_.end(), start_node);
  for (auto it = start_it; *it != end_node; ++it) {
    if (origin_graph->node_set_.find(*it) != origin_graph->node_set_.end()) {
      new_graph->node_set_.insert(*it);
      auto &edges = origin_graph->edge_set_.at(*it);
      for (auto &e : edges) {
        new_graph->edge_set_[*it].emplace_back(e);
        auto r_edge_it = std::find_if(origin_graph->r_edge_set_[e->end_node_].begin(), origin_graph->r_edge_set_[e->end_node_].end(), [&it](EdgeNodePtr &e) {
          return (e->end_node_ == *it);
        });
        new_graph->r_edge_set_[e->end_node_].emplace_back(*r_edge_it);
      }
    }
  }

  new_graph->start_node_ = start_node;
  new_graph->end_node_ = end_node;
  new_graph->node_set_.insert(end_node);
  new_graph->edge_set_[end_node] = std::vector<EdgeNodePtr>();
  new_graph->r_edge_set_[start_node] = std::vector<EdgeNodePtr>();
  CheckIsChain(new_graph);
  return new_graph;
}

void LatencyGraph::ApplyCpsRatioInner(TrieNodePtr root, std::pair<double, double> &qnt_estimator, std::pair<double, double> &source_prof, std::pair<double, double> &avg_prof, std::unordered_set<std::string> &enable_cps_nodes) {
  for (size_t i = 0; i < root->children_.size(); ++i) {
    if (root->children_[i]) {
      ApplyCpsRatioInner(root->children_[i], qnt_estimator, source_prof, avg_prof, enable_cps_nodes);
    }
  }
  if (!root->is_end_) return;

  for (auto &g : root->simple_graphs_) {
    g->cps_time_list_.resize(g->size_list_.size(), 0.0);
    if (is_quant_) {
      for (size_t i = 0; i < g->size_list_.size(); ++i) {
        if (g->size_list_[i] < 0.0) continue;
        // apply qnt ratio & time
        g->cps_time_list_[i] += qnt_estimator.first * g->size_list_[i] + qnt_estimator.second; // us
        g->size_list_[i] /= 4;
        // apply cps ratio & time
        if (is_cps_) {
          if (i == 0) {
            bool enable_cps = (source_prof.first - CPS_RATIO_THRESH) > 1e-6;
            auto ratio = enable_cps ? source_prof.first : 1.0;
            auto time = enable_cps ? source_prof.second : 0.0;
            g->size_list_[i] /= ratio;
            g->cps_time_list_[i] += time;
            if (enable_cps) {
              enable_cps_nodes.insert(g->local_topo_list_[i]->name_);
            }
          } else {
            bool enable_cps = (avg_prof.first - CPS_RATIO_THRESH) > 1e-6;
            auto &node_name = g->local_topo_list_[i]->name_;
            auto node_prim_name = node_name.substr(0, node_name.find_last_of('-'));
            enable_cps &= (allowed_cps_node_names.find(node_prim_name) != allowed_cps_node_names.end());
            
            auto ratio = enable_cps ? avg_prof.first : 1.0;
            auto time = enable_cps ? avg_prof.second : 0.0;
            g->size_list_[i] /= ratio;
            g->cps_time_list_[i] += time;
            if (enable_cps) {
              enable_cps_nodes.insert(g->local_topo_list_[i]->name_);
            }
          }
        }
        g->size_list_[i] /= (1 << 20); // in MB
      }
    } else {
      for (size_t i = 0; i < g->size_list_.size(); ++i) {
        if (g->size_list_[i] < 0.0) continue;
        g->size_list_[i] /= (1 << 20); // in MB
      }
    }
  }
}

void LatencyGraph::PartitionDecision(double bandwidth, double load_factor, PartitionResult &res) {
  PartitionDecisionInner(trie_.root_, bandwidth, load_factor);
  if (is_chain_) {
    // this means the subgraph is a 2-node chain w/o any para paths, just retrieve the results
    res = trie_.root_->children_[0]->part_res_.begin()->second;
  } else {
    // use a fake edge node to compose all parallel chain results
    auto fake_edge = std::make_shared<EdgeNode>();
    for (auto &c : trie_.root_->children_) {
      if (c && c->is_end_) {
        // we must assume there is only one simple graph at 1st level
        fake_edge->subgraphs_.emplace_back(c->simple_graphs_.front());
      }
    }
    res = PartitionResult();
    ComposeMinLatencyForParallelChains(trie_.root_, fake_edge, res, bandwidth, load_factor);
  }
}

// void LatencyGraph::PartitionDecisionInner(TrieNodePtr root, std::vector<int> &cur_pidx, double bandwidth, double load_factor) {
void LatencyGraph::PartitionDecisionInner(TrieNodePtr root, double bandwidth, double load_factor) {
  for (size_t i = 0; i < root->children_.size(); ++i) {
    if (root->children_[i]) {
      PartitionDecisionInner(root->children_[i], bandwidth, load_factor);
    }
  }
  if (root->is_end_) {
    // aggregate results from all children graphs
    for (auto &p : root->complex_edge_part_res_) {
      p.second = PartitionResult();
      ComposeMinLatencyForParallelChains(root, p.first, p.second, bandwidth, load_factor);
    }
    // calculate for the simple graph in this level
    for (auto &g : root->simple_graphs_) {
      FindMinLatencyForChain(root, g, bandwidth, load_factor);
    }
  }
}

void LatencyGraph::ComposeMinLatencyForParallelChains(TrieNodePtr &root, const EdgeNodePtr &edge_for_chains, PartitionResult &res, double bandwidth, double load_factor, bool is_full_explore) {
  MS_EXCEPTION_IF_NULL(edge_for_chains);
  std::unordered_map<std::shared_ptr<LatencyGraph>, PartitionResult*> tmp_res;
  size_t num_para_paths = edge_for_chains->subgraphs_.size();
  size_t num_start_edge_cut = 0;
  for (auto &g : edge_for_chains->subgraphs_) {
    for (auto &c : root->children_) {
      if (c && c->is_end_) {
        auto g_res_it = c->part_res_.find(g);
        if (g_res_it == c->part_res_.end()) continue;
        auto &child_res = g_res_it->second;
        tmp_res[g] = &child_res;

        res.full_local_time_ += child_res.full_local_time_;
        res.full_remote_time_ += child_res.full_remote_time_;
        res.best_cut_time_ += child_res.best_cut_time_;
        res.best_cut_nodes_.insert(child_res.best_cut_nodes_.begin(), child_res.best_cut_nodes_.end());

        // check if the best cut of current graph contains start_edge
        if (child_res.best_cut_nodes_.find(g->start_node_) != child_res.best_cut_nodes_.end()) num_start_edge_cut++;
      }
    }
  }
  auto &g_0 = edge_for_chains->subgraphs_[0];
  // fix res 1: deduct start & end time
  res.full_local_time_ -= (num_para_paths - 1) * (g_0->start_node_->local_exec_time_ + g_0->end_node_->local_exec_time_) / 1e3;
  res.full_remote_time_ -= (num_para_paths - 1) * (load_factor * g_0->start_node_->remote_exec_time_ + load_factor * g_0->end_node_->remote_exec_time_) / 1e3;
  res.best_cut_time_ -= (num_para_paths - 1) * (g_0->start_node_->local_exec_time_ + load_factor * g_0->end_node_->remote_exec_time_) / 1e3;
  // fix res 2: if best cut contains cut edge from the start node, remove redundant trans_time
  if (num_start_edge_cut > 1) {
    auto size_time = FindRealFirstEdgeSizeTime(g_0);
    res.best_cut_time_ -= (num_start_edge_cut - 1) * (1e3 * size_time.first / bandwidth + size_time.second / 1e3);
  }
  // fix res 3: compare res with schemes with multiple start_edge cuts
  if (is_full_explore && num_para_paths > 1) {
    std::vector<std::shared_ptr<LatencyGraph>> cur_subset;
    std::vector<std::shared_ptr<LatencyGraph>> best_cpml_subset;
    double best_time = 1e15;
    CalcFirstEdgeCutSubsets(edge_for_chains->subgraphs_, cur_subset, 0, tmp_res, best_cpml_subset, best_time, bandwidth);
    best_time -= (num_para_paths - 1) * (g_0->start_node_->local_exec_time_ + load_factor * g_0->end_node_->remote_exec_time_) / 1e3;
    res.best_first_edge_cut_time_ = best_time;
    for (auto &g : best_cpml_subset) {
      auto &chain_res = tmp_res[g];
      res.best_first_edge_cut_nodes_.insert(chain_res->best_cut_nodes_.begin(), chain_res->best_cut_nodes_.end());
    }
    res.best_first_edge_cut_nodes_.insert(g_0->start_node_);
    if (best_time < res.best_cut_time_) {
      res.best_cut_time_ = best_time;
      res.best_cut_nodes_.clear();
      res.best_cut_nodes_.insert(res.best_first_edge_cut_nodes_.begin(), res.best_first_edge_cut_nodes_.end());
    }
  } else {
    double all_start_cut_time = 0.0;
    for (auto &p : tmp_res) {
      all_start_cut_time += p.second->best_first_edge_cut_time_;
    }
    all_start_cut_time -= (num_para_paths - 1) * (g_0->start_node_->local_exec_time_ + load_factor * g_0->end_node_->remote_exec_time_) / 1e3;
    auto size_time = FindRealFirstEdgeSizeTime(g_0);
    all_start_cut_time -= (num_para_paths - 1) * (1e3 * size_time.first / bandwidth + size_time.second / 1e3);
    res.best_first_edge_cut_time_ = all_start_cut_time;
    res.best_first_edge_cut_nodes_.insert(g_0->start_node_);
    if (all_start_cut_time < res.best_cut_time_) {
      res.best_cut_time_ = all_start_cut_time;
      res.best_cut_nodes_.clear();
      res.best_cut_nodes_.insert(g_0->start_node_);
    }
  }
}

void LatencyGraph::CalcFirstEdgeCutSubsets(const std::vector<std::shared_ptr<LatencyGraph>>& paths, std::vector<std::shared_ptr<LatencyGraph>>& cur_subset, size_t index, 
                                 std::unordered_map<std::shared_ptr<LatencyGraph>, PartitionResult*> &tmp_res, 
                                 std::vector<std::shared_ptr<LatencyGraph>>& best_cpml_subset, double &best_time, double bandwidth) {
  if (index == paths.size()) {
    if (cur_subset.size() > 1) {
      double total_time = 0.0;
      for (auto &g : cur_subset) {
        total_time += tmp_res.at(g)->best_first_edge_cut_time_;
      }
      size_t num_start_edge_cut = 0;
      std::vector<std::shared_ptr<LatencyGraph>> cmpl_subset;
      for (auto &g : paths) {
        if (std::find(cur_subset.begin(), cur_subset.end(), g) == cur_subset.end()) {
          cmpl_subset.emplace_back(g);
          total_time += tmp_res.at(g)->best_cut_time_;
          if (tmp_res.at(g)->best_cut_nodes_.find(g->start_node_) != tmp_res.at(g)->best_cut_nodes_.end()) num_start_edge_cut++;
        }
      }
      auto &g_0 = cur_subset.front();
      auto size_time = FindRealFirstEdgeSizeTime(g_0);
      total_time -= (cur_subset.size() + num_start_edge_cut - 1) * (1e3 * size_time.first / bandwidth + size_time.second / 1e3);
      if (total_time < best_time) {
        best_time = total_time;
        best_cpml_subset = cmpl_subset;
      }
    }
    return;
  }
  // cur path not included
  CalcFirstEdgeCutSubsets(paths, cur_subset, index + 1, tmp_res, best_cpml_subset, best_time, bandwidth);
  // cur path included
  cur_subset.push_back(paths[index]);
  CalcFirstEdgeCutSubsets(paths, cur_subset, index + 1, tmp_res, best_cpml_subset, best_time, bandwidth);
  cur_subset.pop_back();
}

void LatencyGraph::FindMinLatencyForChain(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor) {
  MS_EXCEPTION_IF_NULL(g);
  if (!g->is_chain_) {
    MS_LOG(EXCEPTION) << "LatencyGraph: cannot find min latency for a non-chain DAG";
  }
  if (g->idx_to_complex_edges_.empty()) {
    // no complex edges in the chain
    FindMinLatencyForChainWithoutCplxEdges(root, g, bandwidth, load_factor);
  } else {
    FindMinLatencyForChainWithCplxEdges(root, g, bandwidth, load_factor);
  }
}

void LatencyGraph::FindMinLatencyForChainWithCplxEdges(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor) {
  // retrieve all complex edges and their result
  std::vector<size_t> cplx_edge_start_idx;
  std::vector<double> cplx_edge_pre_sum;
  std::vector<double> cplx_edge_suf_sum;
  size_t num_cplx_edges = g->idx_to_complex_edges_.size();

  cplx_edge_pre_sum.resize(num_cplx_edges + 1, 0.0);
  cplx_edge_suf_sum.resize(num_cplx_edges + 1, 0.0);
  for (auto &p : g->idx_to_complex_edges_) {
    cplx_edge_start_idx.push_back(p.first);
  }
  for (size_t i = 1; i <= num_cplx_edges; ++i) {
    size_t cur_pre_idx = cplx_edge_start_idx[i - 1];
    size_t cur_suf_idx = cplx_edge_start_idx[num_cplx_edges - i];
    // calculate start_node + end_node time, then deduct from the total sum
    double cur_pre_offset = g->local_topo_list_[cur_pre_idx]->local_exec_time_ + g->local_topo_list_[cur_pre_idx + 1]->local_exec_time_;
    double cur_suf_offset = g->local_topo_list_[cur_suf_idx]->remote_exec_time_ + g->local_topo_list_[cur_suf_idx + 1]->remote_exec_time_;
    cplx_edge_pre_sum[i] = cplx_edge_pre_sum[i - 1] + root->complex_edge_part_res_[g->idx_to_complex_edges_[cur_pre_idx]].full_local_time_ - cur_pre_offset / 1e3; // in ms
    cplx_edge_suf_sum[num_cplx_edges - i] = cplx_edge_suf_sum[num_cplx_edges - i + 1] + root->complex_edge_part_res_[g->idx_to_complex_edges_[cur_suf_idx]].full_remote_time_ - load_factor * cur_suf_offset / 1e3; // in ms
  }

  size_t cplx_idx = 0;
  size_t node_nums = g->local_topo_list_.size();
  double min_cut_time = 1.0e15;
  double min_first_cut_time = 1.0e15;
  size_t min_cut_idx = 0;
  for (size_t i = 0; i < node_nums - 1; ++i) {
    double now_time = 0.0;
    if (g->size_list_[i] < 0) {
      auto &edge = g->idx_to_complex_edges_[i];
      auto &edge_res = root->complex_edge_part_res_[edge];
      // find cut min for children graphs of complex edge
      auto cut_time = edge_res.best_cut_time_ - (g->local_topo_list_[i]->local_exec_time_ + load_factor * g->local_topo_list_[i + 1]->remote_exec_time_) / 1e3;
      now_time = g->local_time_pre_sum_[i + 1] + cut_time + load_factor * g->remote_time_suf_sum_[i + 1];
      now_time += cplx_edge_pre_sum[cplx_idx] + cplx_edge_suf_sum[cplx_idx + 1];
      if (i == 0) {
        auto first_cut_time = edge_res.best_first_edge_cut_time_ - (g->local_topo_list_[i]->local_exec_time_ + load_factor * g->local_topo_list_[i + 1]->remote_exec_time_) / 1e3;
        min_first_cut_time = g->local_time_pre_sum_[i + 1] + first_cut_time + load_factor * g->remote_time_suf_sum_[i + 1];
        min_first_cut_time += cplx_edge_pre_sum[cplx_idx] + cplx_edge_suf_sum[cplx_idx + 1];
      }
      cplx_idx++;
    } else {
      now_time = g->local_time_pre_sum_[i + 1] + (1e3 * g->size_list_[i] / bandwidth + g->cps_time_list_[i] / 1e3) + load_factor * g->remote_time_suf_sum_[i + 1];
      // add time for complex edges (start_node & end_node time deducted)
      now_time += cplx_edge_pre_sum[cplx_idx] + cplx_edge_suf_sum[cplx_idx];
      if (i == 0) {
        min_first_cut_time = now_time;
      }
    }
    if (now_time < min_cut_time) {
      min_cut_time = now_time;
      min_cut_idx = i;
    }
  }
  double full_local_time = g->local_time_pre_sum_.back() + cplx_edge_pre_sum.back();
  double full_remote_time = load_factor * g->remote_time_suf_sum_.front() + cplx_edge_suf_sum.front();

  // write the best cut scheme
  auto res = PartitionResult();
  res.best_cut_time_ = min_cut_time;
  res.full_local_time_ = full_local_time;
  res.full_remote_time_ = full_remote_time;
  if (g->size_list_[min_cut_idx] < 0) {
    auto &edge = g->idx_to_complex_edges_[min_cut_idx];
    auto &edge_res = root->complex_edge_part_res_[edge];
    res.best_cut_nodes_ = edge_res.best_cut_nodes_;
  } else {
    res.best_cut_nodes_.insert(g->local_topo_list_[min_cut_idx]);
  }
  // write the best first-edge cut scheme
  res.best_first_edge_cut_time_ = min_first_cut_time;
  if (g->size_list_[0] < 0) {
    auto &edge = g->idx_to_complex_edges_[0];
    auto &edge_res = root->complex_edge_part_res_[edge];
    res.best_first_edge_cut_nodes_.insert(edge_res.best_first_edge_cut_nodes_.begin(), edge_res.best_first_edge_cut_nodes_.end());
  } else {
    res.best_first_edge_cut_nodes_.insert(g->local_topo_list_[0]);
  }
  root->part_res_[g] = res;
}

void LatencyGraph::FindMinLatencyForChainWithoutCplxEdges(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor) {
  size_t node_nums = g->local_topo_list_.size();
  double min_cut_time = 1.0e15;
  double min_first_cut_time = 1.0e15;
  size_t min_cut_idx = 0;
  for (size_t i = 0; i < node_nums - 1; ++i) {
    double now_time = g->local_time_pre_sum_[i + 1] + (1e3 * g->size_list_[i] / bandwidth + g->cps_time_list_[i] / 1e3) + load_factor * g->remote_time_suf_sum_[i + 1]; 
    if (i == 0) {
      min_first_cut_time = now_time;
    }
    if (now_time < min_cut_time) {
      min_cut_time = now_time;
      min_cut_idx = i;
    }
  }
  double full_local_time = g->local_time_pre_sum_.back();
  double full_remote_time = load_factor * g->remote_time_suf_sum_.front();

  // write the best cut scheme
  auto res = PartitionResult();
  res.best_cut_time_ = min_cut_time;
  res.full_local_time_ = full_local_time;
  res.full_remote_time_ = full_remote_time;
  res.best_cut_nodes_.insert(g->local_topo_list_[min_cut_idx]);
  res.best_first_edge_cut_time_ = min_first_cut_time;
  res.best_first_edge_cut_nodes_.insert(g->local_topo_list_[0]);

  root->part_res_[g] = res;
}

std::pair<double, double> LatencyGraph::FindRealFirstEdgeSizeTime(std::shared_ptr<LatencyGraph> cur_g) {
  MS_EXCEPTION_IF_NULL(cur_g);
  while (cur_g->size_list_.front() < 0) {
    cur_g = cur_g->edge_set_[cur_g->start_node_].front()->subgraphs_.front();
  }
  return std::make_pair(cur_g->size_list_.front(), cur_g->cps_time_list_.front());
}

void LatencyGraph::DrawSimpleGraphs(LatencyGraph::PathIdxTrie &trie, const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }
  ofs << "digraph g {\n";
  ofs << "  fontname = \"Courier New\"\n";
  ofs << "  rankdir = \"LR\"\n";
  ofs << "  node [ fontname = \"Courier New\" ]\n";
  ofs << "  edge [ fontname = \"Courier New\" ]\n";
  ofs << "  graph [ fontsize = 24, spline = true, overlap = false, compound = true];\n";
  ofs << "  ratio = auto;\n\n";

  DrawLatencyGraphAsSubgraph(0, ofs);

  size_t global_counter = 1;
  std::vector<int> cur_pidx;
  std::unordered_map<TrieNodePtr, size_t> counter_map;
  DrawSimpleGraphsInner(trie.root_, cur_pidx, global_counter, counter_map, ofs);

  CostGraph::NodePtr to = nullptr;
  std::string to_suffix;
  
  for (size_t i = 0; i < trie.root_->children_.size(); ++i) {
    if (trie.root_->children_[i]->is_end_) {
      to = trie.root_->children_[i]->simple_graphs_[0]->start_node_;
      ofs << "  \"" << start_node_->name_ << "\" -> \"" << to->name_ << "_" << i << "_0" << "\" [ ltail = \"cluster_0\", lhead = \"cluster_" << counter_map[trie.root_->children_[i]] << "\" ];\n";
    }
  }
  
  ofs << "}";
  ofs.close();
}

size_t LatencyGraph::DrawSimpleGraphsInner(TrieNodePtr root, std::vector<int> &cur_pidx, size_t &global_counter, std::unordered_map<TrieNodePtr, size_t> &counter_map, std::ofstream &ofs) {
  size_t cur_cluster_id = 0;
  if (root->is_end_) {
    // draw cur subgraphs & add to global digraph
    size_t local_counter = 0;
    ofs << "  subgraph \"cluster_" << global_counter << "\" {\n";
    ofs << "    label = \"pidx = ";
    PrintVector(cur_pidx, ofs);
    ofs << "\"\n";
    for (auto &g : root->simple_graphs_) {
      std::ostringstream oss;
      for (auto i : cur_pidx) oss << "_" << i;
      oss << "_" << local_counter;
      auto node_suffix = oss.str();

      ofs << "    subgraph \"cluster_" << global_counter << "_" << local_counter << "\" {\n";
      ofs << "      label = \"\"\n";
      for (auto &node : g->node_set_) {
        MS_EXCEPTION_IF_NULL(node);
        ofs << "      \"" << node->name_ << node_suffix << "\" [ label = \"" << node->name_ << "\", color = dimgray ];\n";
      }
      for (auto &node : g->node_set_) {
        auto &edge_list = g->edge_set_[node];
        for (auto &edge : edge_list) {
          if (!edge->subgraphs_.empty()) {
            ofs << "      \"" << node->name_ << node_suffix << "\" -> \"" << edge->end_node_->name_ << node_suffix << "\" [ color = darkolivegreen, penwidth = 3];\n";
          } else {
            ofs << "      \"" << node->name_ << node_suffix << "\" -> \"" << edge->end_node_->name_ << node_suffix << "\" [ color = dimgray ];\n";
          }
        }
      }
      ofs << "    }\n\n";
      local_counter++;
    }
    ofs << "  }\n\n";
    cur_cluster_id = global_counter++;
    counter_map[root] = cur_cluster_id;
  }
  
  for (size_t i = 0; i < root->children_.size(); ++i) {
    if (root->children_[i]) {
      cur_pidx.push_back(i);
      auto child_cluster_id = DrawSimpleGraphsInner(root->children_[i], cur_pidx, global_counter, counter_map, ofs);
      // add edges between root graph -> children graphs
      CostGraph::NodePtr from = nullptr, to = nullptr;
      std::string from_suffix, to_suffix;
      if (root->children_[i]->is_end_) {
        to = root->children_[i]->simple_graphs_[0]->start_node_;
        std::ostringstream oss;
        for (auto i : cur_pidx) oss << "_" << i;
        to_suffix = oss.str();
      }

      cur_pidx.pop_back();
      
      if (root->is_end_) {
        from = root->simple_graphs_[0]->start_node_;
        std::ostringstream oss;
        for (auto i : cur_pidx) oss << "_" << i;
        from_suffix = oss.str();
      }

      if (from && to) {
        ofs << "  \"" << from->name_ << from_suffix << "_0" <<  "\" -> \"" << to->name_ << to_suffix << "_0" << "\" [ ltail = \"cluster_" << cur_cluster_id << "\", lhead = \"cluster_" << child_cluster_id << "\" ];\n";
      }
    }
  }
  return cur_cluster_id;
}

void LatencyGraphManager::SplitCostGraphIntoLatencyGraphs(bool is_quant, bool is_cps) {
  MS_EXCEPTION_IF_NULL(cost_graph_);
  if (cost_graph_->cut_node_list_.empty()) {
    cost_graph_->GetCutNodes();
  }
  if (cost_graph_->cg_topo_list_.empty()) {
    cost_graph_->TopoSortCostGraph(cost_graph_->cg_topo_list_);
  }
  if (cost_graph_->r_edge_set_.empty()) {
    cost_graph_->ConstructReverseEdges();
  }

  MS_EXCEPTION_IF_CHECK_FAIL((cost_graph_->cut_node_list_.size() > 1), "There should be > 1 cut nodes");
  auto &cut_node_list = cost_graph_->cut_node_list_;
  auto num_cut_nodes = cost_graph_->cut_node_list_.size();
  for (size_t i = 1; i < num_cut_nodes; ++i) {
    auto latency_graph = std::make_shared<LatencyGraph>();
    latency_graph->SetIsQuantCps(is_quant, is_cps);
    latency_graph->ConstructLatencyGraph(cut_node_list[i - 1], cut_node_list[i], cost_graph_);
    latency_graph->MarkLevel();
    latency_graph->ReviseLevel();
    latency_graph->IdentifyPaths();
    latency_graphs_.emplace_back(latency_graph);
  }
  // Apply cps profiles
  ApplyCpsProfile(is_quant, is_cps);

  // prepare pre&suf sum
  local_pre_sum_.resize(num_cut_nodes + 1, 0.0);
  remote_suf_sum_.resize(num_cut_nodes + 1, 0.0);
  cplx_local_pre_sum_.resize(num_cut_nodes, 0.0);
  cplx_remote_suf_sum_.resize(num_cut_nodes, 0.0);
  for (size_t i = 1; i < num_cut_nodes; ++i) {
    local_pre_sum_[i] = local_pre_sum_[i - 1] + latency_graphs_[i - 1]->start_node_->local_exec_time_ / 1e3;
    remote_suf_sum_[num_cut_nodes - i] = remote_suf_sum_[num_cut_nodes - i + 1] + latency_graphs_[num_cut_nodes - i - 1]->end_node_->remote_exec_time_ / 1e3; // in ms
    cplx_local_pre_sum_[i] = cplx_local_pre_sum_[i - 1] + latency_graphs_[i - 1]->GetFullLocalTimeWithoutST() / 1e3; // in ms
    cplx_remote_suf_sum_[num_cut_nodes - i - 1] = cplx_remote_suf_sum_[num_cut_nodes - i] + latency_graphs_[num_cut_nodes - i - 1]->GetFullRemoteTimeWithoutST() / 1e3; // in ms
  }
  local_pre_sum_[num_cut_nodes] = local_pre_sum_[num_cut_nodes - 1] + latency_graphs_.back()->end_node_->local_exec_time_ / 1e3;
  remote_suf_sum_[0] = remote_suf_sum_[1] + latency_graphs_.front()->start_node_->remote_exec_time_ / 1e3;
}

void LatencyGraphManager::LoadCpsProfile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    MS_LOG(EXCEPTION) << "Open file '" << path << "' failed!";
    return;
  }

  std::string line;
  std::getline(ifs, line);
  std::istringstream iss_0(line);
  iss_0 >> qnt_estimator_.first >> qnt_estimator_.second;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    std::string name;
    double ratio, time;
    iss >> name >> ratio >> time;
    cps_profile_[name] = std::make_pair(ratio, time);
  }

  ifs.close();
}

void LatencyGraphManager::DrawLatencyGraphs(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    MS_LOG(WARNING) << "Open file '" << path << "' failed!";
    return;
  }

  if (latency_graphs_.empty()) {
    SplitCostGraphIntoLatencyGraphs(false, false);
  }

  ofs << "digraph g {\n";
  ofs << "  fontname = \"Courier New\"\n";
  ofs << "  rankdir = \"LR\"\n";
  ofs << "  node [ fontname = \"Courier New\" ]\n";
  ofs << "  edge [ fontname = \"Courier New\" ]\n";
  ofs << "  graph [ fontsize = 24, spline = true, overlap = false, compound = true];\n";
  ofs << "  ratio = auto;\n\n";
  for (size_t idx = 0; idx < latency_graphs_.size(); ++idx) {
    latency_graphs_[idx]->DrawLatencyGraphAsSubgraph(idx, ofs);
  }
  ofs << "}";
  ofs.close();
}

size_t LatencyGraphManager::PartitionDecision(double bandwidth, double q_time, double load_factor, LatencyGraph::PartitionResult &part_res) {
  size_t cplx_idx = 0;
  size_t num_lgs = latency_graphs_.size();
  double min_cut_time = 1.0e15;
  size_t min_cut_idx = 0;

  for (size_t i = 0; i < num_lgs; ++i) {
    auto &lg = latency_graphs_[i];
    LatencyGraph::PartitionResult cur_res;
    lg->PartitionDecision(bandwidth, load_factor, cur_res);

    // find cut min for children graphs of complex edge
    double cut_time = cur_res.best_cut_time_ - (lg->start_node_->local_exec_time_ + load_factor * lg->end_node_->remote_exec_time_) / 1e3;
    double now_time = local_pre_sum_[i + 1] + cut_time + load_factor * remote_suf_sum_[i + 1];
    now_time += cplx_local_pre_sum_[cplx_idx] + load_factor * cplx_remote_suf_sum_[cplx_idx + 1];
    cplx_idx++;
    if (now_time < min_cut_time) {
      min_cut_time = now_time;
      min_cut_idx = i;
      part_res = cur_res;
    }
  }

  // only need to add queueing and system overhead time to offloading time once
  min_cut_time += (q_time / 1e3);
  double full_local_time = local_pre_sum_.back() + cplx_local_pre_sum_.back();
  if (full_local_time < min_cut_time) {
    // full local
    part_res.best_cut_time_ = full_local_time;
    part_res.best_cut_nodes_.clear();
    min_cut_idx = latency_graphs_.size();
  }

  return min_cut_idx;
}

CNodePtrList LatencyGraphManager::GenerateKernelGraphSegmentClient(size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes) {
  std::vector<CostGraph::NodePtr> cg_node_list;
  for (size_t i = 0; i < lg_idx; ++i) {
    auto &node_list = latency_graphs_[i]->local_topo_list_;
    std::copy(node_list.begin(), node_list.end() - 1, std::back_inserter(cg_node_list));
  }
  auto &be_cut_lg = latency_graphs_[lg_idx];
  be_cut_lg->GetTopoSortFromCutPointsReverse(cut_nodes, cg_node_list);
  return cost_graph_->GenerateGraphSegment(cg_node_list);
}

CNodePtrList LatencyGraphManager::GenerateKernelGraphSegmentServer(size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes) {
  std::vector<CostGraph::NodePtr> cg_node_list;
  auto &be_cut_lg = latency_graphs_[lg_idx];
  be_cut_lg->GetTopoSortFromCutPoints(cut_nodes, cg_node_list);
  for (size_t i = lg_idx + 1; i < latency_graphs_.size(); ++i) {
    auto &node_list = latency_graphs_[i]->local_topo_list_;
    std::copy(node_list.begin() + 1, node_list.end(), std::back_inserter(cg_node_list));
  }
  return cost_graph_->GenerateGraphSegment(cg_node_list);
}

CNodePtrList LatencyGraphManager::GenerateKernelGraphSegmentBetween(size_t lg_idx_s, const std::unordered_set<CostGraph::NodePtr> &cut_nodes_s, size_t lg_idx_e, const std::unordered_set<CostGraph::NodePtr> &cut_nodes_e, double &base_time) {
  if (lg_idx_s >= lg_idx_e) {
    MS_LOG(EXCEPTION) << "GenerateKernelGraphSegmentBetween: cannot generate segment between lg_idx: " << lg_idx_s << " and " << lg_idx_e;
  }
  std::vector<CostGraph::NodePtr> cg_node_list_s;
  auto &be_cut_lg_s = latency_graphs_[lg_idx_s];
  be_cut_lg_s->GetTopoSortFromCutPoints(cut_nodes_s, cg_node_list_s);

  std::vector<CostGraph::NodePtr> cg_node_list_e;
  if (lg_idx_e < latency_graphs_.size()) {
    auto &be_cut_lg_e = latency_graphs_[lg_idx_e];
    be_cut_lg_e->GetTopoSortFromCutPointsReverse(cut_nodes_e, cg_node_list_e);
  }

  for (size_t i = lg_idx_s + 1; i < lg_idx_e; ++i) {
    auto &node_list = latency_graphs_[i]->local_topo_list_;
    std::copy(node_list.begin() + 1, node_list.end(), std::back_inserter(cg_node_list_s));
  }
  std::copy(cg_node_list_e.begin() + 1, cg_node_list_e.end(), std::back_inserter(cg_node_list_s));
  base_time = std::accumulate(cg_node_list_s.begin(), cg_node_list_s.end(), 0.0, [](double partial_sum, const CostGraph::NodePtr node) {
    return partial_sum + node->local_exec_time_;
  }); // in us
  return cost_graph_->GenerateGraphSegment(cg_node_list_s);
}

void LatencyGraphManager::ApplyCpsProfile(bool is_quant, bool is_cps) {
  if (is_quant && is_cps) {
    if (cps_profile_.empty()) {
      MS_LOG(EXCEPTION) << "LatencyGraphManager: cps profile not found";
    }
    for (size_t i = 0; i < latency_graphs_.size(); ++i) {
      auto &latency_graph = latency_graphs_[i];
      // construct avg cps_ratio&time
      std::pair<double, double> source_prof, avg_prof;
      auto source_prof_it = cps_profile_.find(latency_graph->GetSourceNode()->name_);
      auto sink_prof_it = cps_profile_.find(latency_graph->GetSinkNode()->name_);

      if (source_prof_it != cps_profile_.end() && sink_prof_it != cps_profile_.end()) {
        source_prof = source_prof_it->second;
        avg_prof.first = (source_prof_it->second.first + sink_prof_it->second.first) / 2;
        avg_prof.second = (source_prof_it->second.second + sink_prof_it->second.second) / 2;
      } else if (source_prof_it != cps_profile_.end()) {
        source_prof = source_prof_it->second;
        avg_prof = source_prof;
      } else if (sink_prof_it != cps_profile_.end()) {
        source_prof.first = 1.0;
        source_prof.second = 0.0;
        avg_prof = sink_prof_it->second;
      } else {
        std::pair<double, double> prev_source_prof, next_source_prof;
        auto prev_source_prof_it = cps_profile_.end();
        auto next_source_prof_it = cps_profile_.end();
        // search for prev source node that has profile
        for (int j = i - 1; j >= 0; --j) {
          prev_source_prof_it = cps_profile_.find(latency_graphs_[j]->GetSourceNode()->name_);
          if (prev_source_prof_it != cps_profile_.end()) break;
        }
        // search for next source node that has profile
        for (size_t j = i + 1; j < latency_graphs_.size(); ++j) {
          next_source_prof_it = cps_profile_.find(latency_graphs_[j]->GetSourceNode()->name_);
          if (next_source_prof_it != cps_profile_.end()) break;
        }

        if (prev_source_prof_it != cps_profile_.end() && next_source_prof_it != cps_profile_.end()) {
          avg_prof.first = (prev_source_prof_it->second.first + next_source_prof_it->second.first) / 2;
          avg_prof.second = (prev_source_prof_it->second.second + next_source_prof_it->second.second) / 2;
        } else if (prev_source_prof_it != cps_profile_.end()) {
          avg_prof.first = (prev_source_prof_it->second.first + 1.0) / 2;
          avg_prof.second = (prev_source_prof_it->second.second + 0.0) / 2;
        } else if (next_source_prof_it != cps_profile_.end()) {
          avg_prof.first = (1.0 + next_source_prof_it->second.first) / 2;
          avg_prof.second = (0.0 + next_source_prof_it->second.second) / 2;
        } else {
          MS_LOG(EXCEPTION) << "LatencyGraphManager: cannot find any cps profiles for latency graph " << i;
        }

        source_prof.first = 1.0;
        source_prof.second = 0.0;
      }
      latency_graph->ApplyCpsRatio(qnt_estimator_, source_prof, avg_prof, enable_cps_node_names_);
    }
  } else {
    for (size_t i = 0; i < latency_graphs_.size(); ++i) {
      std::pair<double, double> source_prof, avg_prof;
      source_prof = std::make_pair(0.0, 0.0);
      avg_prof = std::make_pair(0.0, 0.0);
      latency_graphs_[i]->ApplyCpsRatio(qnt_estimator_, source_prof, avg_prof, enable_cps_node_names_);
    }
  }
}

}
}
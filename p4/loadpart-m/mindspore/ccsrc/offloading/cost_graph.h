#ifndef MINDSPORE_OFFLOADING_COST_GRAPH_H
#define MINDSPORE_OFFLOADING_COST_GRAPH_H

#include "utils.h"
#include "status.h"

namespace mindspore {

namespace offloading {

using offloading_serving::GraphProfile;

constexpr double CPS_RATIO_THRESH = 1.1;

std::unordered_map<std::string, float> LoadExecTimeTSV(const std::string &path, size_t &bsz);
void DumpExecTimeTSV(const std::string &path, std::unordered_map<std::string, float>& time_map, size_t bsz);

std::unordered_map<std::string, float> GenerateTimeMapWithRenaming(KernelGraphPtr &g, std::unordered_map<std::string, float> &graph_profile);


class CostGraph {
  friend class LatencyGraph;
  friend class LatencyGraphManager;
  public:
    struct Node {
      size_t id_;
      std::string name_;
      double local_exec_time_;
      double remote_exec_time_;
      bool is_cut_;
      CNodePtrList real_cnodes_;
      std::vector<std::shared_ptr<Node>> inputs_;
      std::vector<size_t> input_sizes_; // only for input parameters and Node outputs, in Byte
      std::vector<size_t> output_sizes_; // only for Node outputs, in Byte
      
      Node() : id_(0), is_cut_(false) { }
      Node(std::string name) : id_(0), name_(name), is_cut_(false) { }
    };
    using NodePtr = std::shared_ptr<Node>;

    struct EdgeEnd {
      NodePtr end_node_;
      size_t trans_size_;

      EdgeEnd(const NodePtr &node, size_t trans_size) : end_node_(node), trans_size_(trans_size) {}
    };
    using EdgeEndPtr = std::shared_ptr<EdgeEnd>;
    using EdgeSet = std::unordered_map<NodePtr, std::vector<EdgeEndPtr>>;

    CostGraph(KernelGraphPtr graph, std::unordered_map<std::string, float> graph_profile) : graph_(graph), graph_profile_(graph_profile) { Construct(); }
    CostGraph(KernelGraphPtr graph, std::string graph_profile_path, double scale_factor = 1.0);
    ~CostGraph() = default;
    
    std::string GetUnifiedOutputName(const CNodePtr& cnode);
    void DrawCostGraph(const std::string &path, bool is_local = true, bool print_shape = false);
    void CostGraphToCSV(const std::string &path);
    void DumpTimeMapToFile(const std::string &path);
    void TopoSortCostGraph(std::vector<NodePtr> &order);
    CNodePtrList GenerateGraphSegment(const std::vector<NodePtr> &node_list);
    GraphId GenerateKernelGraphFromSegment(session::SessionPtr &session_impl, KernelGraphPtr &origin_graph, const CNodePtrList &node_list, std::vector<std::string> &output_name);
    void GetCutNodes();
    void GetNodesByName(const std::vector<std::string> &node_names, std::unordered_set<NodePtr> &nodes);
    void GetNodesByName(const std::unordered_map<std::string, tensor::TensorPtr> &input_map, std::unordered_set<NodePtr> &nodes);
    void SetCloudTimeMap(GraphProfile &remote_time_map, size_t default_bsz = 1);
    std::vector<size_t> GetOutputShape(const NodePtr &node);
    double GetFullLocalTime();

    std::unordered_map<std::string, float>& GetLocalTimeMap() { return local_time_map_; }
    std::unordered_map<std::string, float>& GetGraphProfile() { return graph_profile_; }
    std::unordered_map<std::string, float>& GetCloudTimeMap(size_t bsz);
    KernelGraphPtr& GetFullKernelGraph() { return graph_; }
    NodePtr& GetSourceNode() { return source_node_; }
    NodePtr& GetSinkNode() { return sink_node_; }
  
  private:
    size_t node_num_cnt_ = 0;
    NodePtr source_node_;
    NodePtr sink_node_;
    KernelGraphPtr graph_;
    CNodePtrList topo_list_;
    std::vector<NodePtr> cg_topo_list_;
    std::vector<size_t> cut_node_list_;
    std::unordered_set<NodePtr> node_set_;
    std::unordered_map<std::string, NodePtr> name_node_map_;
    std::unordered_set<NodePtr> input_node_set_;
    EdgeSet edge_set_;
    EdgeSet r_edge_set_;
    std::unordered_map<AnfNodePtr, std::vector<NodePtr>> real_fake_map_;
    std::unordered_map<std::string, float> graph_profile_;
    std::unordered_map<std::string, float> local_time_map_;
    std::unordered_map<size_t, std::unordered_map<std::string, float>> cloud_time_map_;

    std::unordered_map<std::string, int> rename_count_map_;
    std::unordered_map<CNodePtr, CNodePtrList> fused_node_load_node_map_;
    void Construct();
    void ConstructSingleNode(const CNodePtr &cnode, const NodeUsersMap &node_user_map);
    void ConstructFusedNode(const CNodePtr &cnode, const NodeUsersMap &node_user_map);
    void GenerateEdgeSet();
    void ConstructReverseEdges();
    void GenerateSourceSinkNodes(const NodeUsersMap &node_user_map);
    void Tarjan(int clock, const NodePtr &cur_node, const NodePtr &fa_node, std::unordered_map<NodePtr, int> &dfn, std::unordered_map<NodePtr, int> &low);
};

std::string GetCutLabel(const std::unordered_set<CostGraph::NodePtr> &cut_nodes);
std::string GetCutLabel(std::vector<std::string> &cut_nodes_names);

class LatencyGraph {
  friend class LatencyGraphManager;
  public:
    struct EdgeNode {
      CostGraph::NodePtr end_node_;
      size_t trans_size_;
      EdgeNode* r_edge_;
      std::vector<int> level_;
      std::vector<int> pidx_;
      std::vector<std::shared_ptr<LatencyGraph>> subgraphs_;

      EdgeNode() : end_node_(nullptr), r_edge_(nullptr) {}
      EdgeNode(const CostGraph::NodePtr &node, size_t trans_size) : end_node_(node), trans_size_(trans_size), r_edge_(nullptr), level_(std::vector<int>()), pidx_(std::vector<int>()), subgraphs_(std::vector<std::shared_ptr<LatencyGraph>>()) {}
      EdgeNode(const CostGraph::EdgeEndPtr &edge_end) : end_node_(edge_end->end_node_), trans_size_(edge_end->trans_size_), r_edge_(nullptr), level_(std::vector<int>()), pidx_(std::vector<int>()), subgraphs_(std::vector<std::shared_ptr<LatencyGraph>>()) {}
      void SetReverseEdge(std::shared_ptr<EdgeNode> &rp) { r_edge_ = rp.get(); }
    };
    using EdgeNodePtr = std::shared_ptr<EdgeNode>;

    struct PartitionResult {
      std::unordered_set<CostGraph::NodePtr> best_cut_nodes_;
      double full_local_time_;
      double full_remote_time_;
      double best_cut_time_;

      std::unordered_set<CostGraph::NodePtr> best_first_edge_cut_nodes_;
      double best_first_edge_cut_time_;

      PartitionResult() : full_local_time_(0.0), full_remote_time_(0.0), best_cut_time_(0.0) { best_cut_nodes_.clear(); }
    };

    struct PathIdxTrie {
      struct TrieNode {
        bool is_end_ = false;
        TrieNode* parent_ = nullptr;
        std::vector<std::shared_ptr<TrieNode>> children_;
        std::vector<std::shared_ptr<LatencyGraph>> simple_graphs_;
        std::unordered_map<std::shared_ptr<LatencyGraph>, PartitionResult> part_res_; // 3-tuple for current chain
        std::unordered_map<EdgeNodePtr, PartitionResult> complex_edge_part_res_; // 3-tuples for complex edges in current chain
        std::unordered_set<CostGraph::NodePtr> exempt_sta_nodes_;

        TrieNode() : is_end_(false), parent_(nullptr), children_(std::vector<std::shared_ptr<TrieNode>>()), simple_graphs_(std::vector<std::shared_ptr<LatencyGraph>>()) {}
      };
      PathIdxTrie() { root_ = std::make_shared<TrieNode>(); }
      ~PathIdxTrie() = default;
      void Insert(const std::vector<int> &pidx);

      std::shared_ptr<TrieNode> root_;
    };
    using TrieNodePtr = std::shared_ptr<PathIdxTrie::TrieNode>;

    LatencyGraph() = default;
    ~LatencyGraph() = default;
    void ConstructLatencyGraph(size_t start_idx, size_t end_idx, std::shared_ptr<CostGraph> &cost_graph);
    void MarkLevel();
    void ReviseLevel();
    void IdentifyPaths();
    void ApplyCpsRatio(std::pair<double, double> &qnt_estimator, std::pair<double, double> &source_prof, std::pair<double, double> &avg_prof, std::unordered_set<std::string> &enable_cps_nodes);
    double GetFullLocalTimeWithoutST();
    double GetFullRemoteTimeWithoutST();
    void PartitionDecision(double bandwidth, double load_factor, PartitionResult &res);
    void GetTopoSortFromCutPoints(const std::unordered_set<CostGraph::NodePtr> &cut_nodes, std::vector<CostGraph::NodePtr> &res);
    void GetTopoSortFromCutPointsReverse(const std::unordered_set<CostGraph::NodePtr> &cut_nodes, std::vector<CostGraph::NodePtr> &res);
    void DrawLatencyGraph(const std::string &path);
    void DrawLatencyGraphAsSubgraph(size_t idx, std::ofstream &ofs);

    void SetIsQuantCps(bool is_quant, bool is_cps) { is_quant_ = is_quant, is_cps_ = is_cps; }
    CostGraph::NodePtr &GetSourceNode() { return start_node_; }
    CostGraph::NodePtr &GetSinkNode() { return end_node_; }

  private:
    bool is_chain_;
    bool is_quant_;
    bool is_cps_;
    CostGraph::NodePtr start_node_;
    CostGraph::NodePtr end_node_;
    std::unordered_set<CostGraph::NodePtr> node_set_;
    std::unordered_map<CostGraph::NodePtr, std::vector<EdgeNodePtr>> edge_set_;
    std::unordered_map<CostGraph::NodePtr, std::vector<EdgeNodePtr>> r_edge_set_;
    
    std::vector<CostGraph::NodePtr> local_topo_list_;
    std::vector<double> local_time_pre_sum_;
    std::vector<double> remote_time_suf_sum_;
    std::vector<double> size_list_;
    std::vector<double> cps_time_list_;

    std::map<size_t, EdgeNodePtr> idx_to_complex_edges_;
    
    PathIdxTrie trie_;

    int GetLevel(EdgeNodePtr &e);
    int GetLevel(EdgeNode *e);
    void UpdateLevelVector(EdgeNodePtr &e);
    void CheckIsChain(std::shared_ptr<LatencyGraph> &g);
    void ConstructTimeArrays(std::shared_ptr<LatencyGraph> &g);
    void MarkPathByOrder(std::vector<EdgeNodePtr> &out_edges, int last_level, std::vector<int> cur_pidx);
    void ConstructSimpleGraphs(std::map<std::vector<int>, std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>>> &tmp_path_set);
    void ConstructSimpleGraphsInner(TrieNodePtr root, std::vector<int> &cur_pidx, const std::map<std::vector<int>, std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>>> &tmp_path_set);
    void ConstructUpperLevelSimpleGraph(TrieNodePtr &root, const std::vector<int> &cur_pidx, const std::vector<std::pair<CostGraph::NodePtr, EdgeNodePtr>> &edges, std::vector<std::shared_ptr<LatencyGraph>> &children_graphs, std::vector<std::shared_ptr<LatencyGraph>> &cur_node_graphs, std::unordered_set<CostGraph::NodePtr> &exempt_sta);
    void SplitSimpleGraph(std::shared_ptr<LatencyGraph> &origin_graph, std::vector<std::shared_ptr<LatencyGraph>> &split_graphs, 
                          std::unordered_set<CostGraph::NodePtr> &tmp_start_nodes, std::unordered_set<CostGraph::NodePtr> &tmp_end_nodes, std::unordered_set<CostGraph::NodePtr> &sta_nodes);
    std::shared_ptr<LatencyGraph> ConstructGraphFromGraph(const std::shared_ptr<LatencyGraph> &origin_graph, const CostGraph::NodePtr &start_node, const CostGraph::NodePtr &end_node);
    void ApplyCpsRatioInner(TrieNodePtr root, std::pair<double, double> &qnt_estimator, std::pair<double, double> &source_prof, std::pair<double, double> &avg_prof, std::unordered_set<std::string> &enable_cps_nodes);

    void PartitionDecisionInner(TrieNodePtr root, double bandwidth, double load_factor);
    void ComposeMinLatencyForParallelChains(TrieNodePtr &root, const EdgeNodePtr &edge_for_chains, PartitionResult &res, double bandwidth, double load_factor, bool is_full_explore = false);
    void CalcFirstEdgeCutSubsets(const std::vector<std::shared_ptr<LatencyGraph>>& paths, std::vector<std::shared_ptr<LatencyGraph>>& cur_subset, size_t index, 
                                 std::unordered_map<std::shared_ptr<LatencyGraph>, PartitionResult*> &tmp_res, 
                                 std::vector<std::shared_ptr<LatencyGraph>>& best_cpml_subset, double &best_time, double bandwidth);
    void FindMinLatencyForChain(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor);
    void FindMinLatencyForChainWithCplxEdges(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor);
    void FindMinLatencyForChainWithoutCplxEdges(TrieNodePtr &root, std::shared_ptr<LatencyGraph> &g, double bandwidth, double load_factor);
    std::pair<double, double> FindRealFirstEdgeSizeTime(std::shared_ptr<LatencyGraph> cur_g);
    void DrawSimpleGraphs(PathIdxTrie &trie, const std::string &path);
    size_t DrawSimpleGraphsInner(TrieNodePtr root, std::vector<int> &cur_pidx, size_t &global_counter, std::unordered_map<TrieNodePtr, size_t> &counter_map, std::ofstream &ofs);
};

class LatencyGraphManager {
  public:
    LatencyGraphManager() = default;
    ~LatencyGraphManager() = default;
    void SetCostGraph(std::shared_ptr<CostGraph>& cost_graph) { cost_graph_ = cost_graph; }
    void SplitCostGraphIntoLatencyGraphs(bool is_quant, bool is_cps);
    void LoadCpsProfile(const std::string& path);
    void DrawLatencyGraphs(const std::string &path);
    size_t PartitionDecision(double bandwidth, double q_time, double load_factor, LatencyGraph::PartitionResult &part_res);
    CNodePtrList GenerateKernelGraphSegmentClient(size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes);
    CNodePtrList GenerateKernelGraphSegmentServer(size_t lg_idx, const std::unordered_set<CostGraph::NodePtr> &cut_nodes);
    CNodePtrList GenerateKernelGraphSegmentBetween(size_t lg_idx_s, const std::unordered_set<CostGraph::NodePtr> &cut_nodes_s, size_t lg_idx_e, const std::unordered_set<CostGraph::NodePtr> &cut_nodes_e, double &base_time);
    size_t GetTotalLatencyGraphNums() { return latency_graphs_.size(); }
    bool CheckEnableCps(const std::string& name) { return (enable_cps_node_names_.find(name) != enable_cps_node_names_.end()); }
  private:
    std::shared_ptr<CostGraph> cost_graph_ = nullptr;
    std::vector<std::shared_ptr<LatencyGraph>> latency_graphs_;
    std::vector<double> local_pre_sum_;
    std::vector<double> remote_suf_sum_;
    std::vector<double> cplx_local_pre_sum_;
    std::vector<double> cplx_remote_suf_sum_;

    // cps and quant profiles
    std::pair<double, double> qnt_estimator_; // ax + b (x in B, y in us)
    std::unordered_map<std::string, std::pair<double, double>> cps_profile_; // ratio, time (us)
    std::unordered_set<std::string> enable_cps_node_names_;

    void ApplyCpsProfile(bool is_quant, bool is_cps);
};

}
}

#endif
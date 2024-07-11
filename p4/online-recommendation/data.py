import mindspore as ms
import networkx as nx
import mindspore_gl
import numpy as np

def dblp_publication_venue_to_feature(venue):

    features = [0] * 6

    AI_keys = ["Artificial Intelligence", "Machine Learning"]
    Algorithm_keys = ["Theory", "Algorithms"]
    CV_keys = ["Compute Vision"]
    Database_keys = ["Database"]
    Datamine_keys = ["Data Mining"]
    IR_keys = ["Infomation Retrieval"]

    for key in AI_keys:
        if key in venue:
            features[0] = 1
    
    for key in Algorithm_keys:
        if key in venue:
            features[1] = 1

    for key in CV_keys:
        if key in venue:
            features[2] = 1
    
    for key in Database_keys:
        if key in venue:
            features[3] = 1
    
    for key in Datamine_keys:
        if key in venue:
            features[4] = 1

    for key in IR_keys:
        if key in venue:
            features[5] = 1

    return np.array(features)

def dblp_year_to_time(year):
    start_year = 2000
    if year >= start_year:
        return (year - start_year)//2
    else:
        return 0

class DynamicGraphData:

    def __init__(self, name):
        self.max_time = 100

        self.nodes = []
        self.node_types = 1
        self.edges = []
        self.time_edges =[]

        self.name = name
        self.time_range = 10
        self.nx_net = nx.Graph()
        
        #train data
        self.type_nodes_feature_cnt = []
        self.type_nodes_feature_matrix = []

        self.train_node_cnt = 1000
        self.train_nodes = []
        self.train_features = None 
        self.train_edges = []

        self.nodes_neighbors = []
        self.nodes_sampled_metapaths = []
        
        #for dblp datasets
        self.articles = {}
        self.authors = {}
        self.venues = {}


    def read_data(self):
        if self.name == "dblp":
            with open("data/dblp/outputacm.txt", "r") as f:
                content = f.read()
            articles = content.split("\n\n")

            self.node_types = 3

            start_year = 2000
            end_year = 2018
            self.time_range = (end_year-start_year)//2 + 1
            for i in range(self.time_range):
                self.time_edges.append([])

            for bi in articles:
                line = bi.split("\n")
                if len(line[0]) > 0 and line[0][0] != '#':
                    line = line[1:]
                if len(line) < 5:
                    continue
                
                title, authors, year, pub_venue, index = line[0], line[1], line[2], line[3], line[4]

                if index.startswith("#index") and len(index) > 6:
                    article_index = int(index[6:])
                features = dblp_publication_venue_to_feature(pub_venue)

                if year.startswith("#t") and len(year) > 2:
                    time = dblp_year_to_time(int(year[2:]))
                
                author_index_list = []
                if authors.startswith("#@") and len(authors) > 2:
                    author_list = authors[2:].split(",")
                else:
                    author_list = set()
                for author in author_list:
                    if author in self.authors:
                        i = self.authors[author]
                        self.nodes[i][1]["feature"] = np.bitwise_and(self.nodes[i][1]["feature"], features)
                        author_index_list.append(i)
                    else:
                        i = len(self.nodes)
                        self.nodes.append((i, {"type": 1, "feature": feature}))
                        self.authors[author] = i
                        author_index_list.append(i)

                if pub_venue in self.venues:
                    venue_index = self.venues[pub_venue]
                else:
                    venue_index = len(self.nodes)
                    self.nodes.append((i, {"type": 2, "feature": feature}))
                    self.venues[pub_venue] = i

                article_index = len(self.nodes)
                self.nodes.append((i, {"type": 0, "feature": feature}))

                for i in author_index_list:
                    self.edges.append((i, article_index))
                    self.time_edges[time].append((i, article_index))
                self.edges.append((venue_index, article_index))
                self.time_edges[time].append((venue_index, article_index))

#               self.article_nodes.append({"title": title, "time": time, "lable": lable, "authors": author_index_list, "venue": venue_index})
            
            self.type_nodes_feature_cnt = [6, 6, 6]
            self.type_nodes_feature_matrix = [None] * 3
        elif self.name == "epinions":
            pass
        elif self.name == "aminer":
            pass

    def print_data_info(self):
        print(self.time_range)
#        for time in range(len(self.time_edges)):
#            print(time, len(self.time_edges[time]))

        label_count_dic = {}
        for node in self.nodes:
            if node[1]["label"] in label_count_dic:
                label_count_dic[node[1]["label"]] += 1
            else:
                label_count_dic[node[1]["label"]] = 1
        print(label_count_dic)


    def construct_nx_network(self):
        self.nx_net.add_nodes_from(self.nodes)
        self.nx_net.add_edges_from(self.time_edges[0])    

        print("number of nodes t0:", nx.number_of_nodes(self.nx_net))
        print("number of edges t0:", nx.number_of_edges(self.nx_net))            

    def find_neighbours_in_h_hoop(self, v):
        q = Queue()
        q.put((v[0],0))
        visited = set()
        visited.add(v[0])
        while not q.empty():
            v, distance = q.get()
            n_list = nx.neighbors(self.graph, v)
            for node in n_list:
                if node not in visited and distance < self.h:
                    q.put((node, distance+1))
                    visited.add(node)
        visited.remove(v)
        print(v, len(visited))
        return visited

    def find_metapaths(self):
        for node in self.nodes:
            neighbors = self.find_neighbours_in_h_hoop(node)


    def prepare_train_data(self):
        
        for node in self.train_nodes:
            self.type_nodes_feature_cnt[node[1]["type"]] = max(len(node[1]["feature"]), self.type_nodes_feature_cnt[node[1]["type"]])
            if self.type_nodes_feature_matrix[node[1]["type"]]:
                np.vstack(self.type_nodes_feature_matrix[node[1]["type"]], node[1]["feature"])
            else:
                self.type_nodes_feature_matrix[node[1]["type"]] = node[1]["feature"]
            


if __name__ == "__main__":
    dblp_data = DynamicGraphData("dblp")
    dblp_data.read_data()
    dblp_data.print_data_info()
    dblp_data.construct_nx_network()
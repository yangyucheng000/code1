import mindspore as ms
from data import DynamicGraphData
from model import *

if __name__ == "__main__":
    #dataset parameters
    datasets = ["dblp"]

    #model parameters
    model_args = {}
    model_args["d"] = 64
    model_args["h"] = 2
    model_args["metapath_cutoff"] = 3
    model_args["relu_alpha"] = 0.01
    model_args["he"] = 3
    model_args["ha"] = 3
    model_args["r"] = 0.5
    model_args["re"] = 0.5
    model_args["ra"] = 0.5

    #
    if args["GPU"]:
        context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)
    if args["fuse"]:
        context.set_context(device_target="GPU", mode=context.GRAPH_MODE, enable_graph_kernel=True)

    for dataname in datasets:
        data = DynamicGraphData(dataname)
        data.read_data()
        data.print_data_info()
        data.construct_nx_network()
    
        #train_data, validation_data, test_data = data.make_small_sample_dataset()

        model = DHANE_Model(data, model_args)
        model.train()
        model.test(test_classifier, test_data, test_label)
        model.dynamic_embedding_update()
        model.test(test_classifier, test_data, test_label)
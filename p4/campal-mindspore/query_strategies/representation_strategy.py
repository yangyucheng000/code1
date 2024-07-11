from .strategy import Strategy
from .builder import STRATEGIES
import mindspore


@STRATEGIES.register_module()
class RepresentationStrategy(Strategy):
    def __init__(self, dataset, net, args, logger, timestamp):
        super(RepresentationStrategy, self).__init__(dataset, net, args, logger, timestamp)
        assert self.args.aug_metric_ulb in ['standard', 'chamfer', 'hausdorff']

    def calculating_embeddings(self, dataset):
        raise NotImplementedError

    def calculating_dist_matrix(self, dataset_1, dataset_2):
        # Just for density calculation
        eps = 1e-3
        aug_times = 1 + self.args.aug_ratio_ulb + self.args.mix_ratio_ulb
        embeddings_1 = self.calculating_embeddings(dataset_1).reshape([aug_times, len(dataset_1) // aug_times, -1]).swapaxes(0, 1)
        embeddings_2 = self.calculating_embeddings(dataset_2).reshape([aug_times, len(dataset_2) // aug_times, -1]).swapaxes(0, 1)
        embedding_reordered_1 = embeddings_1.reshape([aug_times, len(dataset_1) // aug_times, -1]).swapaxes(0, 1).reshape([len(dataset_1), -1])
        embeddings_reordered_2 = embeddings_2.reshape([aug_times, len(dataset_2) // aug_times, -1]).swapaxes(0, 1).reshape([len(dataset_2), -1])
        dist_mat_full = mindspore.ops.cdist(embedding_reordered_1, embeddings_reordered_2, 2.0).reshape(
            [len(dataset_1) // aug_times, aug_times, len(dataset_2) // aug_times, aug_times]).swapaxes(1, 2)
        dist_mat_full += 1e-3
        if self.args.aug_metric_ulb == 'standard':
            return mindspore.ops.min(mindspore.ops.min(dist_mat_full, axis=2)[0], axis=2)[0]
        elif self.args.aug_metric_ulb == 'chamfer':
            return mindspore.ops.mean(mindspore.ops.min(dist_mat_full, axis=2)[0], axis=2) + \
                   mindspore.ops.mean(mindspore.ops.min(dist_mat_full, axis=3)[0], axis=2)
        elif self.args.aug_metric_ulb == 'hausdorff':
            return mindspore.ops.max(mindspore.ops.max(mindspore.ops.min(dist_mat_full, axis=2)[0], axis=2)[0],
                            mindspore.ops.max(mindspore.ops.min(dist_mat_full, axis=3)[0], axis=2)[0])
        else:
            raise NotImplementedError

    def query(self, n, aug_args_list=None):
        raise NotImplementedError

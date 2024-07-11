from .strategy import Strategy
from .builder import STRATEGIES
from datasets.dataloader import GetHandler
import mindspore

@STRATEGIES.register_module()
class ScoreStrategy(Strategy):
    def __init__(self, dataset, net, args, logger, timestamp):
        super(ScoreStrategy, self).__init__(dataset, net, args, logger, timestamp)
        assert self.args.aug_metric_ulb in ['normal', 'max', 'min', 'sum', 'density']
        self.sim_mat = None

    def calculating_scores(self, dataset_u):
        raise NotImplementedError

    def calculating_sim_matrix(self, dataset_u):
        # Just for density calculation
        aug_times = 1 + self.args.aug_ratio_ulb + self.args.mix_ratio_ulb
        sim_factor = 2.0
        dataset_u_original = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform)
        embeddings_ori = self.get_embedding(self.clf, dataset_u_original)
        embeddings_all = self.get_embedding(self.clf, dataset_u).reshape(
            [1 + self.args.aug_ratio_ulb + self.args.mix_ratio_ulb, len(dataset_u) // aug_times, -1])
        dist_mat = mindspore.ops.sqrt(mindspore.ops.sum(mindspore.ops.pow(embeddings_all - embeddings_ori, 2), dim=2))
        self.sim_mat = mindspore.ops.exp(-1.0 * dist_mat / sim_factor)

    def aggregate_scores(self, scores):
        if (self.args.aug_ratio_ulb + self.args.mix_ratio_ulb == 0) or (not self.args.aug_ulb_on):
            return scores
        scores = scores.reshape([1 + self.args.aug_ratio_ulb + self.args.mix_ratio_ulb,
                                 len(self.dataset.DATA_INFOS['train_u'])])
        if self.args.aug_metric_ulb in ['normal', 'max']:
            return mindspore.ops.max(scores, axis=0)[0]
        elif self.args.aug_metric_ulb == 'sum':
            return mindspore.ops.sum(scores, axis=0)
        elif self.args.aug_metric_ulb == 'min':
            return mindspore.ops.min(scores, axis=0)[0]
        elif self.args.aug_metric_ulb == 'density':
            return mindspore.ops.mean(scores * self.sim_mat, axis=0)
        else:
            raise NotImplementedError

    def query(self, n, aug_args_list=None):
        dataset_u = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform, 1,
                               self.args.aug_ratio_ulb, self.args.mix_ratio_ulb, self.args.aug_ulb_evaluation_mode,
                               strength_mode=self.args.aug_ulb_strength_mode, args_list=aug_args_list)
        if self.args.aug_metric_ulb == 'density':
            self.calculating_sim_matrix(dataset_u)
        aggre_scores = self.aggregate_scores(self.calculating_scores(dataset_u))
        return aggre_scores.sort()[1][-n:]

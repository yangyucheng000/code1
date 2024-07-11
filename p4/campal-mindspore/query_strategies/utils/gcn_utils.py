# import numpy as np
# import torch
from sklearn.metrics import pairwise_distances
import abc

import mindspore
import mindspore.numpy as np
from mindspore import Tensor


# class SubsetSequentialSampler(torch.utils.data.Sampler):
class SubsetSequentialSampler(mindspore.dataset.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def select_batch_unc_(self, **kwargs):
        return self.select_batch_unc_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):
    def __init__(self, X, metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            x = self.features[cluster_centers]
            # Update min_distances for all examples given new cluster center.
            dist = pairwise_distances(self.features, x, metric=self.metric)  # ,n_jobs=4)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            print('Getting transformed features...')
            #   self.features = model.transform(self.X)
            print('Calculating distances...')
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    # lnl = torch.log(scores[lbl])
    lnl = mindspore.ops.log(scores[lbl])
    # lnu = torch.log(1 - scores[nlbl])
    lnu = mindspore.ops.log(1 - scores[nlbl])
    # labeled_score = torch.mean(lnl)
    labeled_score = mindspore.ops.mean(lnl)
    # unlabeled_score = torch.mean(lnu)
    unlabeled_score = mindspore.ops.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss


# def aff_to_adj(x, y=None):
#     x = x.detach().cpu().numpy()
#     adj = np.matmul(x, x.transpose())
#     adj += -1.0 * np.eye(adj.shape[0])
#     adj_diag = np.sum(adj, axis=0)  # row-wise sum
#     adj = np.matmul(adj, np.diag(1 / adj_diag))
#     adj = adj + np.eye(adj.shape[0])
#     adj = torch.Tensor(adj).cuda()
#     return adj
def aff_to_adj(x, y=None):
    x = x.asnumpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # row-wise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = Tensor(adj)
    return adj


def get_kcg(models, labeled_data_size, unlabeled_loader, args, SUBSET):
    models['backbone'].eval()
    # features = torch.tensor([]).cuda()
    features = Tensor([])

    # with torch.no_grad():
    #     for inputs, _, _ in unlabeled_loader:
    #         # inputs = inputs.cuda()
    #         _, features_batch, _ = models['backbone'](inputs)
    #         features = torch.cat((features, features_batch), 0)
    #     feat = features.detach().cpu().numpy()
    #     new_av_idx = np.arange(SUBSET, (SUBSET + labeled_data_size))
    #     sampling = kCenterGreedy(feat)
    #     batch = sampling.select_batch_(new_av_idx, args.num_query)
    #     other_idx = [x for x in range(SUBSET) if x not in batch]
    # return other_idx + batch

    for inputs, _, _ in unlabeled_loader:
        # inputs = inputs.cuda()
        _, features_batch, _ = models['backbone'](inputs)
        features = mindspore.ops.cat((features, features_batch), 0)
    feat = features.asnumpy()
    new_av_idx = np.arange(SUBSET, (SUBSET + labeled_data_size))
    sampling = kCenterGreedy(feat)
    batch = sampling.select_batch_(new_av_idx, args.num_query)
    other_idx = [x for x in range(SUBSET) if x not in batch]
    return other_idx + batch



def get_features(model, unlabeled_loader):
    model.eval()
    # features = torch.tensor([]).cuda()
    features = Tensor([])
    # with torch.no_grad():
    #     for inputs, _, _, _ in unlabeled_loader:
    #         inputs = inputs.cuda()
    #         _, features_batch, _ = model(inputs)
    #         features = torch.cat((features, features_batch), 0)
    #     feat = features  # .detach().cpu().numpy()

    for inputs, _, _, _ in unlabeled_loader:
        # inputs = inputs.cuda()
        _, features_batch, _ = model(inputs)
        features = mindspore.ops.cat((features, features_batch), 0)
    feat = features  # .detach().cpu().numpy()
    return feat

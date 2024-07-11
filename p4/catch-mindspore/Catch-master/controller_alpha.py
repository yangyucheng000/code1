import copy

import numpy as np
# import torch.nn.functional as F
# from torch.distributions import Categorical, Bernoulli
import torch.nn as nn
import torch
from constant import Operation
import random
from random import choice
from collections import OrderedDict

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import mindspore.nn.probability.distribution as msd
from mindspore.common.initializer import One

# torch.set_num_threads(5)
ms.context.set_context(runtime_num_threads = 5)

# ms.set_context(mode=ms.GRAPH_MODE)
# ms.set_context(save_graphs=2, save_graphs_path="./ir")
# ms.set_context(reserve_class_name_in_scope=False)

class Controller(nn.Cell):
    """
    Policy Network
    """

    def __init__(self, args):
        super(Controller, self).__init__()
        self.args = args
        self.hidden_size = 300
        self.embedding_size = 8
        self.rnn_cycle_num = 2
        self.with_combine_search = True
        self.with_hyper_param_search = False
        # self.device = torch.device('cuda', args.cuda) if torch.cuda.is_available() and args.cuda != -1  else torch.device('cpu')
        #ms
        self.device = ms.set_context(device_target='GPU')
        # =======================================================
        self.continuous_col = args.continuous_col
        self.discrete_col = args.discrete_col
        self.binned_continuous_col = \
            [col_name for col_name in self.continuous_col]

        if not self.with_combine_search:
            self.combine_col = []
        elif self.args.task_type == 'regression':
            self.combine_col = self.discrete_col
        elif self.args.task_type == 'classifier':
            self.combine_col = self.binned_continuous_col + self.discrete_col
        else:
            raise ValueError('Value type error, check task_type')

        # print(len(self.continuous_col), len(self.discrete_col))
        # self.all_columns = args.continuous_col + args.discrete_col
        self.ops_type = Operation.ops_type

        #
        self.fe_action_list = self.continuous_col + self.combine_col
        self.fe_action_num = len(self.fe_action_list)

        #
        continuous_action_space = []
        continuous_action_space.extend(Operation.value_ops)
        for c_fe in self.continuous_col:
            col_actions = [(math, c_fe) for math in Operation.math_ops]
            continuous_action_space.extend(col_actions)

        #
        discrete_action_space = []
        for fe in self.combine_col:
            discrete_action_space.append(('combine', fe))
        # print('discrete_action_space', discrete_action_space)
        # print('continuous_action_space', continuous_action_space)

        #
        c_action_space_len = len(continuous_action_space)
        d_action_space_len = len(discrete_action_space)

        #
        max_action_space_len = max(c_action_space_len, d_action_space_len)
        # append_none_num = abs(c_action_space_len - d_action_space_len)
        # print('append_none_num', append_none_num)

        candidate_d_space = \
            ([None] + discrete_action_space) * max_action_space_len
        handled_discrete_space = candidate_d_space[0:max_action_space_len]

        candidate_c_space = \
            ([None] + continuous_action_space) * max_action_space_len
        handled_continuous_space = candidate_c_space[0:max_action_space_len]

        #
        # if c_action_space_len > d_action_space_len:
        #     handled_discrete_space = \
        #         discrete_action_space + [None]*append_none_num
        #     handled_continuous_space = continuous_action_space
        # elif c_action_space_len < d_action_space_len:
        #     handled_discrete_space = discrete_action_space
        #     handled_continuous_space = \
        #         continuous_action_space + [None]*append_none_num
        # else:
        #     handled_discrete_space = discrete_action_space
        #     handled_continuous_space = continuous_action_space
        # print('handled_discrete_space', len(handled_discrete_space),
        #       handled_discrete_space)
        # print('handled_continuous_space', len(handled_continuous_space),
        #       handled_continuous_space)

        #
        self.need_embedding_actions = \
            continuous_action_space + discrete_action_space + \
            Operation.special_ops + Operation.ops_type

        print('need_embedding_actions', len(self.need_embedding_actions),
              self.need_embedding_actions)

        #
        self.handled_space_len = len(handled_discrete_space)
        self.handled_continuous_space = handled_continuous_space
        self.handled_discrete_space = handled_discrete_space

        #
        # =======================================================
        # handled_space_len = len(handled_discrete_space)
        # print('handled_space_len', handled_space_len)

        merge_action_space = []
        for i in range(self.handled_space_len):
            merge_action_space.extend(Operation.ops_type)
            if len(merge_action_space) > self.handled_space_len:
                break
        self.merge_action_space = merge_action_space[0:self.handled_space_len]

        # print('merge_action_space', len(self.merge_action_space),
        #       self.merge_action_space)
        # exit()
        # =======================================================

        #
        # =======================================================
        model_param_dict = Operation.rf_classification_param
        self.model_param_col = list(model_param_dict.keys())
        self.model_param_num = len(self.model_param_col)
        self.param_action_space = OrderedDict()
        for param_key, param_value in model_param_dict.items():
            single_param_space = param_value * self.handled_space_len
            single_param_space = single_param_space[0:self.handled_space_len]
            self.param_action_space[param_key] = single_param_space

        # print('param_action_space', self.param_action_space)
        # print('model_param_col', self.model_param_col)
        # =======================================================

        # Embedding
        # =======================================================
        self.embedding = nn.Embedding(
            len(self.need_embedding_actions), self.embedding_size)
        # =======================================================

        # RNN
        # =======================================================
        #  LSTMCell
        self.otp_rnn = nn.LSTMCell(
            2 * self.embedding_size * self.fe_action_num, self.hidden_size)

        #  LSTMCell
        self.ops_rnn = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # M operation embedding
        #ms:nn.Linear改为了Dense
        self.opt_decoder = nn.Dense(
            self.hidden_size, self.fe_action_num * self.handled_space_len)

        # T operation embedding
        self.ops_decoder = nn.Dense(
            self.hidden_size, self.fe_action_num * self.handled_space_len)

        #
        if self.with_hyper_param_search:
            self.hyper_param_decoder = nn.Dense(
                self.hidden_size, self.model_param_num * self.handled_space_len)
        else:
            self.hyper_param_decoder = None

        #
        state_size = (1, self.hidden_size)

#         self.init_ops_h = nn.Parameter(torch.zeros(state_size))
#         self.init_ops_c = nn.Parameter(torch.zeros(state_size))

#         self.init_otp_h = nn.Parameter(torch.ones(state_size))
#         self.init_otp_c = nn.Parameter(torch.ones(state_size))
        
        #ms
        zeros = ops.Zeros()
        a = zeros(state_size, ms.float32)
        self.init_ops_h = Parameter(a)
        self.init_ops_c = Parameter(a)
        self.init_otp_h = Parameter(a)
        self.init_otp_c = Parameter(a)
        # =======================================================


        self.init_parameters()

        # self.handled_space_len = handled_space_len
        # self.handled_continuous_space = handled_continuous_space
        # self.handled_discrete_space = handled_discrete_space
        # print(self.fe_action_num, self.handled_space_len)
    
    #ms:forward->construct
    def construct(self, input_data, h_c_t_list, is_hyper_param=False):
        #ms
        # net = nn.Embedding(use_one_hot=True,vocab_size=self.hidden_size, embedding_size=self.embedding_size,dtype=ms.int32)
        # print(self.embedding)
        # breakpoint()
        input_data = self.embedding(input_data)
        
        #torch
        # input_data = self.embedding(input_data)
        # print('input_data', input_data)
        
        input_data = input_data.reshape(
            -1, 2 * self.fe_action_num * self.embedding_size)
        # RNN
        otp_h_t, otp_c_t = self.otp_rnn(input_data, h_c_t_list[0])
        ops_h_t, ops_c_t = self.ops_rnn(otp_h_t, h_c_t_list[1])

        if is_hyper_param:
            hyper_param_logits = self.hyper_param_decoder(ops_h_t)
            return hyper_param_logits

        otp_logits = self.opt_decoder(otp_h_t)
        ops_logits = self.ops_decoder(ops_h_t)

        res_h_c_t_list = [(otp_h_t, otp_c_t), (ops_h_t, ops_c_t)]
        return res_h_c_t_list, ops_logits, otp_logits

    def sample_cycle_component(self, logits, output_type, random_ratio=0):
        #
        softmax1 = ops.Softmax()
        probs = softmax1(logits)
        action_index_local = msd.Categorical(probs=probs, dtype=ms.int32).sample()
        # action_index_local = Categorical(logits=logits).sample()

        # prob_matrix = F.softmax(logits, dim=1)
        softmax2 = ops.Softmax(axis=1)
        prob_matrix = softmax2(logits)
        # log_prob_matrix = F.log_softmax(logits, dim=1)
        log_softmax = nn.LogSoftmax(axis=1)
        log_prob_matrix = log_softmax(logits)
        
        # print('action_index_local', action_index_local)
        # print('prob_matrix', prob_matrix.shape, prob_matrix)

        base_p = 5 / self.handled_space_len
        p_max = prob_matrix.max(axis=1)[0]
        # prob_mean = prob_matrix.mean(dim=1)
        # prob_std = prob_matrix.std(dim=1)
        # prob_cov = prob_std / prob_mean
        # print('：', prob_cov)

        action_choice_list = [i for i in range(logits.shape[1])]
        # print('action_choice_list', action_choice_list)
        # print('action_index_local', action_index_local)
        for i in range(action_index_local.shape[0]):
            random_prod = random.uniform(0, 1)
            # if random_prod > 1-random_ratio:
            if (random_prod > 1 - random_ratio) and (p_max[i] > base_p):
                action_index_local[i] = choice(action_choice_list)
        # print('action_index_local', action_index_local)
        
        reshape_action_index = action_index_local.reshape(-1, 1)
        
        # reshape_action_index = reshape_action_index.asnumpy()
        # reshape_action_index = reshape_action_index.astype(np.int64)
        
        # prob_vec = torch.gather(
#             prob_matrix, dim=1,
#             index=reshape_action_index).reshape(1, -1)

#         log_prob_vec = torch.gather(
#             log_prob_matrix, dim=1,
#             index=reshape_action_index).reshape(1, -1)
        
        # print("reshape_action_index ",reshape_action_index)        
        # reshape_action_index = Tensor(reshape_action_index,ms.int32)
        # prob_matrix = prob_matrix.asnumpy()
        # print("prob_matrix ",prob_matrix)
        # print("reshape_action_index ",reshape_action_index)
        # breakpoint()

        
        #ms
        # num = np.zeros(reshape_action_index.shape,dtype=np.float32)
        # for i in range(reshape_action_index.shape[0]):
        #     for j in range(reshape_action_index.shape[1]):
        #         a = reshape_action_index[i][j]
        #         print("a",a)
        #         breakpoint()
        #         num[i][j] = prob_matrix[i][a]
        # prob_vec = Tensor(num, dtype=ms.float32).reshape(1,-1)
        # print("my_prob_vec",prob_vec)        
        # breakpoint()
        
        op = ops.Cast()
        reshape_action_index = op(reshape_action_index,ms.int32)
        
        prob_vec = ops.GatherD()(
            prob_matrix, 1,
            reshape_action_index).reshape(1, -1)

        log_prob_vec = ops.GatherD()(
            log_prob_matrix, 1,
            reshape_action_index).reshape(1, -1)

        # action_entropy = -1 * torch.sum(prob_vec * log_prob_vec, dim=1)
        #ms
        op = ops.ReduceSum(keep_dims=False)
        action_entropy = -1 * op(prob_vec * log_prob_vec, 1)

        # ====================================================
        if output_type == 'ops_type':
            action_name = \
                [self.merge_action_space[i] for i in action_index_local]
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)
        elif output_type == 'hyper_param':
            action_name = dict()
            for idx, p_value in enumerate(action_index_local):
                single_col_name = self.model_param_col[idx]
                single_action_space = self.param_action_space[single_col_name]
                action_name[single_col_name] = single_action_space[p_value]
            action_index_global = None
            # print('action_name', action_name)

        elif output_type == 'ops':
            c_index_local = action_index_local[0:len(self.continuous_col)]
            d_index_local = action_index_local[len(self.continuous_col):]
            c_action_name = \
                [self.handled_continuous_space[i] for i in c_index_local]
            d_action_name = \
                [self.handled_discrete_space[i] for i in d_index_local]

            action_name = c_action_name + d_action_name
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)
        else:
            action_name = None
            action_index_global = None
            print('er')
        # print('action_name', action_name)
        # ====================================================

        return {
            #ms:detach()->asnumpy()
            'ops_type': output_type,
            'prob_vec': Tensor(prob_vec.asnumpy(),ms.float32),
            'action_entropy': Tensor(action_entropy.asnumpy(),ms.float32),
            'action_index_local': Tensor(action_index_local.asnumpy(),ms.float32),
            'action_index_global': action_index_global,
            'action_name': action_name
        }

    @staticmethod
    def prod_cycle_component(logits, action_index_local):

        # prob_matrix = F.softmax(logits, dim=1)
        softmax2 = ops.Softmax(axis=1)
        prob_matrix = softmax2(logits)
        # log_prob_matrix = F.log_softmax(logits, dim=1)
        log_softmax = nn.LogSoftmax(axis=1)
        log_prob_matrix = log_softmax(logits)

        reshape_action_index = action_index_local.reshape(-1, 1)
        
        # prob_vec = torch.gather(
        #     prob_matrix, dim=1,
        #     index=reshape_action_index).reshape(1, -1)
        # log_prob_vec = torch.gather(
        #     log_prob_matrix, dim=1,
        #     index=reshape_action_index).reshape(1, -1)
        
        #ms
        # reshape_action_index = reshape_action_index.asnumpy()
        # reshape_action_index = reshape_action_index.astype(np.int64)
        # print("reshape_action_index ",reshape_action_index)
        # reshape_action_index = Tensor(reshape_action_index,ms.int32)
        # print("reshape_action_index ",reshape_action_index)
        # breakpoint()
        
        op = ops.Cast()
        reshape_action_index = op(reshape_action_index,ms.int32)
        
        prob_vec = ops.GatherD()(
            prob_matrix, 1,
            reshape_action_index).reshape(1, -1)
        # print(prob_vec)
        # breakpoint()

        log_prob_vec = ops.GatherD()(
            log_prob_matrix, 1,
            reshape_action_index).reshape(1, -1)

        # action_entropy = -1 * torch.sum(prob_vec * log_prob_vec, dim=1)
        #ms
        op = ops.ReduceSum(keep_dims=False)
        action_entropy = -1 * op(prob_vec * log_prob_vec, 1)

        return {
            'prob_vec': prob_vec,
            'action_entropy': action_entropy
        }

    def infer_cycle_component(self, logits, output_type):
        # prob_matrix = F.softmax(logits, dim=1)
        softmax2 = ops.Softmax(axis=1)
        prob_matrix = softmax2(logits)
        # log_prob_matrix = F.log_softmax(logits, dim=1)
        log_softmax = nn.LogSoftmax(axis=1)
        log_prob_matrix = log_softmax(logits)

        prob_vec, action_index_local = prob_matrix.max(1)
        # prob_vec = prob_vec.unsqueeze(0)
        #ms
        expand_dims = ops.ExpandDims()
        prob_vec = expand_dims(prob_vec,0)
        # print('infer', prob_vec, action_index_local)

        if output_type == 'ops_type':
            action_name = \
                [self.merge_action_space[i] for i in action_index_local]
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)
        elif output_type == 'hyper_param':
            action_name = dict()
            for idx, p_value in enumerate(action_index_local):
                single_col_name = self.model_param_col[idx]
                single_action_space = self.param_action_space[single_col_name]
                action_name[single_col_name] = single_action_space[p_value]
            action_index_global = None
        elif output_type == 'ops':
            c_index_local = action_index_local[0:len(self.continuous_col)]
            d_index_local = action_index_local[len(self.continuous_col):]
            c_action_name = \
                [self.handled_continuous_space[i] for i in c_index_local]
            d_action_name = \
                [self.handled_discrete_space[i] for i in d_index_local]

            action_name = c_action_name + d_action_name
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)

        else:
            action_name = None
            action_index_global = None
            print('er')

        return {
            #ms
            'action_index_local': Tensor(action_index_local.asnumpy(),ms.float32),
            'action_index_global': action_index_global,
            'action_name': action_name,
            'prob_vec': Tensor(prob_vec.asnumpy(),ms.float32)
        }

    def sample(self):
        #S0
        input_data = ['PADDING'] * 2 * self.fe_action_num
        input_data = [self.need_embedding_actions.index(i) for i in input_data]
        
        # input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
        #ms
        expand_dims = ops.ExpandDims()
        input_data = expand_dims(Tensor(input_data,dtype=ms.int64),0)
        
        # print('input_data', input_data)

        hc_list = self.init_rnn_hidden()   # T-h0, M-h0

        sample_history = []
        for pos in range(self.rnn_cycle_num):
            res_h_c_t_list, ops_logits, otp_logits = self.construct(
                input_data, hc_list)
            otp_logits = otp_logits.view(
                self.fe_action_num, self.handled_space_len)
            ops_logits = ops_logits.view(
                self.fe_action_num, self.handled_space_len)

            # decoder-->softmaxa-->sample to actions_id
            opt_info = self.sample_cycle_component(otp_logits, 'ops_type')
            # print('opt_info', opt_info)

            ops_info = self.sample_cycle_component(ops_logits, 'ops')
            # print('ops_info', ops_info)

            cycle_info = [opt_info, ops_info]

            # tensor([28, 20, 36, 16, 35, 42, 30, 15, 11])
            # action_index_global = ops_info['action_index_global']
            action_index_global = \
                [info['action_index_global'] for info in cycle_info]
            # action_index_global = torch.cat(action_index_global, 0)
            #ms
            action_index_global = ops.concat(action_index_global, 0)

            action_index_local = \
                [info['action_index_local'] for info in cycle_info]

            # tensor([[0.0238, 0.0233, .... 0.0234]])
            prob_vec = [info['prob_vec'] for info in cycle_info]
            
            # prob_vec = torch.cat(prob_vec, 1)
            #ms
            prob_vec = ops.concat(prob_vec, 1)

            # tensor([1.5761])
            action_entropy = \
                [info['action_entropy'] for info in cycle_info]
            # action_entropy = torch.cat(action_entropy, 0)
            # action_entropy = torch.sum(action_entropy).unsqueeze(0)
            
            #ms
            action_entropy = ops.concat(action_entropy, 0)
            op = ops.ReduceSum(keep_dims=False)
            
            expand_dims = ops.ExpandDims()
            action_entropy = expand_dims(op(action_entropy),0)

            opt_actions_name = opt_info['action_name']
            ops_actions_name = ops_info['action_name']

            trans_actions = self.actions_trans(
                ops_actions_name, opt_actions_name)

            sample_history.append({
                'action_index_global': action_index_global,
                'action_index_local': action_index_local,
                'trans_actions': trans_actions,
                'action_entropy': action_entropy,
                'prob_vec': prob_vec,
                #ms
                'otp_logits': Tensor(otp_logits.asnumpy(),ms.float32),
                'ops_logits': Tensor(ops_logits.asnumpy(),ms.float32)
            })
            hc_list = res_h_c_t_list
            # input_data = action_index_global.unsqueeze(0).to(self.device)
            #ms
            expand_dims = ops.ExpandDims()
            input_data = expand_dims(action_index_global,0)

        return_dict = dict()
        return_dict['feature_engineering'] = sample_history
        # ====================================================
        if self.with_hyper_param_search:
            hyper_param_logits = self.construct(
                input_data, hc_list, is_hyper_param=True)
            hyper_param_logits = hyper_param_logits.view(
                self.model_param_num, self.handled_space_len)
            hyper_param_info = self.sample_cycle_component(
                hyper_param_logits, 'hyper_param')
            # hyper_param_action = opt_info['action_name']
            return_dict['hyper_parameter'] = hyper_param_info
        else:
            return_dict['hyper_parameter'] = None
        # ====================================================
        return return_dict

    def get_prod(self, sample_history, detach_bool=False):
        input_data = ['PADDING'] * 2 * self.fe_action_num
        input_data = [self.need_embedding_actions.index(i) for i in input_data]
        # input_data = torch.LongTensor(input_data).unsqueeze(0).to(self.device)
        #ms
        expand_dims = ops.ExpandDims()
        input_data = expand_dims(Tensor(input_data),0)

        sample_history_fe = sample_history['feature_engineering']
        sample_history_hp = sample_history['hyper_parameter']

        hc_list = self.init_rnn_hidden()

        prod_history = []
        for sample_index, sampled_his in enumerate(sample_history_fe):
            res_h_c_t_list, ops_logits, otp_logits = self.construct(
                input_data, hc_list)

            otp_logits = otp_logits.view(
                self.fe_action_num, self.handled_space_len)
            ops_logits = ops_logits.view(
                self.fe_action_num, self.handled_space_len)

            action_index_local = sampled_his['action_index_local']
            opt_info = self.prod_cycle_component(
                otp_logits, action_index_local[0])
            ops_info = self.prod_cycle_component(
                ops_logits, action_index_local[1])
            # print('opt_info', opt_info)
            # print('ops_info', ops_info)

            cycle_info = [opt_info, ops_info]

            # tensor([[0.0238, 0.0233, .... 0.0234]])
            prob_vec = [info['prob_vec'] for info in cycle_info]
            # prob_vec = torch.cat(prob_vec, 1)
            #ms
            prob_vec = ops.concat(prob_vec, 1)

            action_entropy = \
                [info['action_entropy'] for info in cycle_info]
            # action_entropy = torch.cat(action_entropy, 0)
            # action_entropy = torch.sum(action_entropy).unsqueeze(0)
            #ms
            action_entropy = ops.concat(action_entropy, 0)
            op = ops.ReduceSum(keep_dims=False)
            action_entropy = expand_dims(op(action_entropy),0)

            if detach_bool:
                prod_history.append({
                    #ms
                    'action_entropy': Tensor(action_entropy.asnumpy(),ms.float32),
                    'prob_vec': Tensor(prob_vec.asnumpy(),ms.float32)
                })
            else:
                prod_history.append({
                    'action_entropy': action_entropy,
                    'prob_vec': prob_vec
                })
            hc_list = res_h_c_t_list
            # input_data = sampled_his['action_index_global'].unsqueeze(0).to(self.device)
            #ms
            expand_dims = ops.ExpandDims()
            input_data = expand_dims(sampled_his['action_index_global'],0)

        return_dict = dict()
        return_dict['feature_engineering'] = prod_history
        # ====================================================
        if self.with_hyper_param_search:
            hyper_param_logits = self.construct(
                input_data, hc_list, is_hyper_param=True)
            hyper_param_logits = hyper_param_logits.view(
                self.model_param_num, self.handled_space_len)

            hp_action_index_local = sample_history_hp['action_index_local']
            hyper_param_info = self.prod_cycle_component(
                hyper_param_logits, hp_action_index_local)

            return_dict['hyper_parameter'] = hyper_param_info
        else:
            return_dict['hyper_parameter'] = None
        # ====================================================
        return return_dict
    def infer_trans(self, prob_matrix, output_type):
        prob_vec, action_index_local = prob_matrix.max(1)
        # prob_vec = prob_vec.unsqueeze(0)
        #ms
        expand_dims = ops.ExpandDims()
        prob_vec = expand_dims(prob_vec,0)
        # print('infer', prob_vec, action_index_local)

        if output_type == 'ops_type':
            action_name = \
                [self.merge_action_space[i] for i in action_index_local]
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)
        elif output_type == 'hyper_param':
            action_name = dict()
            for idx, p_value in enumerate(action_index_local):
                single_col_name = self.model_param_col[idx]
                single_action_space = self.param_action_space[single_col_name]
                action_name[single_col_name] = single_action_space[p_value]
            action_index_global = None
        elif output_type == 'ops':
            c_index_local = action_index_local[0:len(self.continuous_col)]
            d_index_local = action_index_local[len(self.continuous_col):]
            c_action_name = \
                [self.handled_continuous_space[i] for i in c_index_local]
            d_action_name = \
                [self.handled_discrete_space[i] for i in d_index_local]

            action_name = c_action_name + d_action_name
            action_index_global = \
                [self.need_embedding_actions.index(i) for i in action_name]
            # action_index_global = torch.LongTensor(action_index_global)
            #ms
            action_index_global = Tensor(action_index_global)

        else:
            action_name = None
            action_index_global = None
            print('error')

        return {
            #ms
            'action_index_local': Tensor(action_index_local.asnumpy(),ms.float32),
            'action_index_global': action_index_global,
            'action_name': action_name,
            'prob_vec': Tensor(prob_vec.asnumpy(),ms.float32)
        }

    def inference(self):
        input_data = ['PADDING'] * 2 * self.fe_action_num
        input_data = [self.need_embedding_actions.index(i) for i in input_data]
        # input_data = torch.LongTensor(input_data).unsqueeze(0)
        #ms
        expand_dims = ops.ExpandDims()
        input_data = expand_dims(Tensor(input_data),0)
        
        hc_list = self.init_rnn_hidden()

        infer_history = []
        for pos in range(self.rnn_cycle_num):
            res_h_c_t_list, ops_logits, otp_logits = self.construct(
                input_data, hc_list)
            otp_logits = otp_logits.view(
                self.fe_action_num, self.handled_space_len)
            ops_logits = ops_logits.view(
                self.fe_action_num, self.handled_space_len)

            opt_info = self.infer_cycle_component(otp_logits, 'ops_type')
            ops_info = self.infer_cycle_component(ops_logits, 'ops')

            # *************************************************************
            cycle_info = [opt_info, ops_info]

            action_index_global = \
                [info['action_index_global'] for info in cycle_info]
            # action_index_global = torch.cat(action_index_global, 0)
            #ms
            action_index_global = ops.concat(action_index_global, 0)

            action_index_local = \
                [info['action_index_local'] for info in cycle_info]

            # tensor([[0.0238, 0.0233, .... 0.0234]])
            prob_vec = [info['prob_vec'] for info in cycle_info]
            # prob_vec = torch.cat(prob_vec, 1)
            #ms
            prob_vec = ops.concat(prob_vec, 1)

            ops_actions_name = ops_info['action_name']
            opt_actions_name = opt_info['action_name']
            trans_actions = self.actions_trans(
                ops_actions_name, opt_actions_name)

            # *************************************************************
            infer_history.append({
                'action_index_global': action_index_global,
                'action_index_local': action_index_local,
                'trans_actions': trans_actions,
                'prob_vec': prob_vec,
                #ms
                'otp_logits': Tensor(otp_logits.asnumpy(),ms.float32),
                'ops_logits': Tensor(ops_logits.asnumpy(),ms.float32)
            })

            hc_list = res_h_c_t_list
            # input_data = action_index_global.unsqueeze(0)
            #ms
            expand_dims = ops.ExpandDims()
            input_data = expand_dims(action_index_global,0)
            # if ops_type_info['action_name'] == 'replace_END' or \
            #         ops_type_info['action_name'] == 'concat_END':
            #     break

        return_dict = dict()
        return_dict['feature_engineering'] = infer_history
        # *************************************************************
        if self.with_hyper_param_search:
            hyper_param_logits = self.construct(
                input_data, hc_list, is_hyper_param=True)
            hyper_param_logits = hyper_param_logits.view(
                self.model_param_num, self.handled_space_len)
            hyper_param_info = self.infer_cycle_component(
                hyper_param_logits, 'hyper_param')
            return_dict['hyper_parameter'] = hyper_param_info
        else:
            return_dict['hyper_parameter'] = None
        # *************************************************************

        return return_dict

    def actions_trans(self, ops_actions_name, opt_actions_name):
        # print('ops_actions_name', ops_actions_name)
        # print('opt_actions_name', opt_actions_name)
        trans_actions = []
        for idx, val in enumerate(ops_actions_name):
            col_name = self.fe_action_list[idx]
            col_opt_name = opt_actions_name[idx]
            if val is None:
                temp_action = [col_name, 'None', col_opt_name]
                trans_actions.append(temp_action)
                continue

            # if isinstance(val, tuple) and val[0] is None:


            if isinstance(val, tuple) and val[0] in Operation.math_ops:
                # if col_name != val[1]:
                temp_action = [col_name, val[1], val[0], col_opt_name]
                trans_actions.append(temp_action)
            elif isinstance(val, str) and val in Operation.value_ops:
                temp_action = [col_name, val, col_opt_name]
                trans_actions.append(temp_action)
            elif isinstance(val, tuple) and val[0] == 'combine':
                if col_name != val[1]:
                    temp_action = [col_name, val[1], val[0], col_opt_name]
                    trans_actions.append(temp_action)

        return trans_actions

    def init_rnn_hidden(self):
        state_size = (1, self.hidden_size)
        
        #ms:expand->broadcast_to
        # init_otp_h = self.init_otp_h.broadcast_to(*state_size).contiguous()
        shape = (1,self.hidden_size)
        init_otp_h = self.init_otp_h.broadcast_to(shape).contiguous()
        init_otp_c = self.init_otp_c.broadcast_to(shape).contiguous()

        init_ops_h = self.init_ops_h.broadcast_to(shape).contiguous()
        init_ops_c = self.init_ops_c.broadcast_to(shape).contiguous()

        h_c_t_list = [(init_otp_h, init_otp_c), (init_ops_h, init_ops_c)]

        return h_c_t_list

    def init_parameters(self):
        init_range = ms.tensor(0.1)
        param:ms.Parameter
        for name, param in self.parameters_and_names(): 
            param.set_data(ops.uniform(param.shape, -init_range, init_range))
        
        # self.ops_decoder.bias.data.fill_(0)
        # self.opt_decoder.bias.data.fill_(0)
        # if self.with_hyper_param_search:
        #     self.hyper_param_decoder.bias.data.fill_(0)
        fill = ops.Fill()    
        self.ops_decoder.bias.set_data(fill(ms.float32,self.ops_decoder.bias.shape,0))
        self.opt_decoder.bias.set_data(fill(ms.float32,self.opt_decoder.bias.shape,0))
        if self.with_hyper_param_search:
            self.hyper_param_decoder.bias.set_data(fill(ms.float32,self.hyper_param_decoder.bias.shape,0))


if __name__ == '__main__':
    load_args = 'fertility_Diagnosis'
    from main import args, dataset_config

    search_dataset_info = dataset_config[load_args]
    for key, value in search_dataset_info.items():
        setattr(args, key, value)

    controller = Controller(args)
    print('controller', controller)
    # sample test
    sample_his = controller.sample()
    # print('sample_his', len(sample_his), sample_his)
    print('sample feature_engineering', sample_his['feature_engineering'])
    print('sample_actions',
          [i['trans_actions'] for i in sample_his['feature_engineering']])
    print('sample hyper_parameter', sample_his['hyper_parameter'])

    # prod test
    prod_his = controller.get_prod(sample_his)
    print('prod_his', len(prod_his['feature_engineering']), prod_his)

    # inference test
    infer_his = controller.inference()
    print('infer_his', len(infer_his['feature_engineering']), infer_his)

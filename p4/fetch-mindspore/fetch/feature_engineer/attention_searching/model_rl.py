import mindspore.nn as nn
import mindspore
from .modules_rl import StatisticLearning, EncoderLayer, SelectOperations, ReductionDimension, weight_init
import logging, os
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore import load_checkpoint, load_param_into_net

class Actor(nn.Cell):
    def __init__(self, args, data_nums, operations, d_model, d_k, d_v, d_ff, n_heads, dropout=None, enc_load_pth=None):
        super(Actor, self).__init__()
        self.args = args
        self.reduction_dimension = ReductionDimension(data_nums, d_model)
        self.encoder = EncoderLayer(d_model, d_k, d_v, d_ff, n_heads, dropout)
        logging.info(f"Randomly initial encoder")
        if os.path.exists(enc_load_pth):
            self.encoder.load_state_dict(mindspore.load(enc_load_pth))
            logging.info(f"Successfully load encoder, enc_load_pth:{enc_load_pth}")
        self.select_operation = SelectOperations(d_model, operations)
        self.c_nums = len(args.c_columns)
        # print("------------")
        # print(data_nums)
        # print(operations)
        # print(d_model)
        # print(self.c_nums)
        # print("------------")
        self.layernorm = nn.LayerNorm(normalized_shape=(data_nums,))
        # print("Actor initialize done")

    # def forward(self, input, step):
    def construct(self, input, step):
        # print("Actor Construct begin")
        input_norm = self.layernorm(input)
        data_reduction_dimension = self.reduction_dimension(input_norm)
        # print("###")
        data_reduction_dimension = mindspore.ops.where(mindspore.ops.isnan(data_reduction_dimension),
                                               mindspore.ops.full_like(data_reduction_dimension, 0), data_reduction_dimension)
        # print("###")
        encoder_output = self.encoder(data_reduction_dimension)
        # print("###")
        encoder_output = mindspore.ops.where(mindspore.ops.isnan(encoder_output), mindspore.ops.full_like(encoder_output, 0), encoder_output)
        # print("###")
        output = self.select_operation(encoder_output)
        output = mindspore.ops.where(mindspore.ops.isnan(output), mindspore.ops.full_like(output, 0), output)
        operation_softmax = mindspore.ops.softmax(output, axis=-1)
        
        # print("Actor Construct done")
        
        return operation_softmax, data_reduction_dimension.squeeze(), \
               encoder_output.squeeze(), output

#         # 使用MindSpore处理NaN的情况
#         isnan = P.IsNan()(data_reduction_dimension)
#         zeros_like = P.ZerosLike()(data_reduction_dimension)
#         data_reduction_dimension = P.Select()(isnan, zeros_like, data_reduction_dimension)
        
#         encoder_output = self.encoder(data_reduction_dimension)
        
#         # 使用MindSpore处理NaN的情况
#         isnan = P.IsNan()(encoder_output)
#         zeros_like = P.ZerosLike()(encoder_output)
#         encoder_output = P.Select()(isnan, zeros_like, encoder_output)
        
#         output = self.select_operation(encoder_output)
        
#         # 使用MindSpore处理NaN的情况
#         isnan = P.IsNan()(output)
#         zeros_like = P.ZerosLike()(output)
#         output = P.Select()(isnan, zeros_like, output)
        
#         operation_softmax = nn.Softmax(axis=-1)(output)
        
#         return operation_softmax, P.Squeeze(-1)(data_reduction_dimension), \
#                P.Squeeze(-1)(encoder_output), output
        

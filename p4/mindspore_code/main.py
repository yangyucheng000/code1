from option import args
import torch
import utility
#import data
import model
import loss
from trainer import Trainer
import os
from model.blindsr import make_model
from mindspore import context
from getdata import get_loader
from mindspore.communication.management import init
from mindspore.context import ParallelMode


if __name__ == '__main__':
    checkpoint = utility.checkpoint(args)
    device_id = int(os.getenv('DEVICE_ID', '0'))

    device_num = int(os.getenv('RANK_SIZE', '1'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)#context.GRAPH_MODE

    if checkpoint.ok:
        loader =get_loader(args) 
        model = make_model(args)
        t = Trainer(args, loader, model, checkpoint)
        t.train()
        checkpoint.done()

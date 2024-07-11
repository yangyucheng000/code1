from test_opt import args
import utility
#import data
from getdata import get_loader
import model
import loss
from trainer import Trainer
import os 
from model.blindsr import make_model
from mindspore import context

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mindspore.communication.management import init
from mindspore.context import ParallelMode


if __name__ == '__main__':
    checkpoint = utility.checkpoint(args)


def write_log_flie(txt_name,add_log):
        with open('log/' + txt_name,'a') as f:
                f.write(add_log)


def model_test(args):
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader =get_loader(args) 

        model_1 =make_model(args)
        t = Trainer(args, loader, model_1, checkpoint)
        t.test()

    checkpoint.done()

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False, device_id=device_id)
    parml_list =[0,1.2,2.4,3.6]
    dataset_list = ['Set5','Set14','BSD100','Urban100']
    list_index = 0
    for j in dataset_list:
        args.data_test = j
        for i in parml_list:
            args.sig = i
            args.test_noise = 10
            model_test(args)


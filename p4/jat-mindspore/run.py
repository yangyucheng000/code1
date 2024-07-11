import argparse
import mindspore as ms

from mindformers import Trainer, TrainingArguments

def main(run_mode='train',
         task='text_generation',
         model_type='gpt2',
         pet_method='',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='hello!'):
    # 环境初始化
    ms.set_context(device_target="GPU")
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=8, learning_rate=0.001, warmup_steps=100,
                                      sink_mode=True, sink_size=2)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task,
                   model=model_type,
                   pet_method=pet_method,
                   args=training_args,
                   train_dataset=train_dataset,
                   eval_dataset=eval_dataset)
    if run_mode == 'finetune':
        task.finetune()
    elif run_mode == 'eval':
        task.evaluate(eval_checkpoint=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='finetune', required=False, help='set run mode for model.')
    parser.add_argument('--task', default='text_classification', required=False, help='set task type.')
    parser.add_argument('--model_type', default='txtcls_bert_base_uncased', required=False, help='set model type.')
    parser.add_argument('--train_dataset', default="glue_data/CoLA/train/", help='set train dataset.')
    parser.add_argument('--eval_dataset', default="glue_data/CoLA/dev/", help='set eval dataset.')
    parser.add_argument('--predict_data', default='hello!', help='input data used to predict.')
    parser.add_argument('--pet_method', default='', help="set finetune method, now support type: ['', 'lora']")
    args = parser.parse_args()
    main(run_mode=args.run_mode,
         task=args.task,
         model_type=args.model_type,
         pet_method=args.pet_method,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data)
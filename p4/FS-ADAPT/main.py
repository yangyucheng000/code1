import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops
from networks import *
from losses import *
from mindspore import context 
import datasets
import time

if __name__ == "__main__":
    
    #mindspore configuration
    context.set_context(mode=context.PYNATIVE_MODE)
    
    max_train_tuples = 100
    
    loss_margin = 1
    lamda = 1

    #set parameters
    total_echos = 5
    learning_rate = 0.001
    batch_size = 1
    max_batch = 200

    train_dataset, test_dataset, online_test_dataset_normal, online_test_dataset_anomaly = datasets.get_data()
    input_shape = datasets.get_input_shape()

    net = FsAdaptNet(input_shape)
    net.set_grad() 
    optimizer = nn.Adam(net.trainable_params(), learning_rate)
    model = ms.Model(net, optimizer=optimizer) 

    """
    test one round
    batch = train_dataset.batch(batch_size=1)
    x0, x1, x2, x3, y0, y1 = next(batch.create_tuple_iterator())
    train_output = net(x0, x1, x2, x3, y0, y1)
    print(train_output)
    exit()
    """
    # training
    def forward_fn(x0, x1, x2, x3, y0, y1):
        output, loss = net(x0, x1, x2, x3, y0, y1)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(x0, x1, x2, x3, y0, y1):
        loss, grads = grad_fn(x0, x1, x2, x3, y0, y1)
        optimizer(grads)
        return loss

    def train(model, dataset):
        batch = 0
        net.set_train()
        for data in dataset:
            loss = train_step(data[0], data[1], data[2], data[3], data[4], data[5])
            if batch % 2 == 0:
                loss, current = loss.asnumpy(), batch
                print("batch:", batch , "      loss:", loss)
            batch += 1
            if batch > max_batch:
                return

    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)
    for t in range(total_echos):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, train_dataset)
    print("Train Done!")  

    #offline test
    def test(model, dataset):
        dataset = dataset.batch(batch_size=1)
        net.set_train(False)
        right = 0
        total = 0
        for data in dataset:
            x_test = net.feature_extractor(ms.Tensor(data[0], ms.float32))
            x_positive = net.feature_extractor(ops.expand_dims(ms.Tensor(datasets.get_random_positive_sample(), ms.float32), 0))
            x_negative = net.feature_extractor(ops.expand_dims(ms.Tensor(datasets.get_random_negative_sample(), ms.float32), 0))
            output, _ = net.domain_classifier(x_test, x_positive, x_negative, 0, 1)
            if output == data[2][0]:
                right += 1
            total += 1
        print("right/total:", right, "/", total, "=", right/total)
    test(model, test_dataset)


    #online test
    def online_test(model, dataset):
        dataset = dataset.batch(batch_size=batch_size)
        net.set_train(False)
        right = 0
        total = 0
        for data in dataset:
            x_test = net.feature_extractor(ms.Tensor(data[0], ms.float32))
            s_pos = datasets.get_random_positive_sample()
            s_neg = datasets.get_random_negative_sample()
            x_positive = net.feature_extractor(ops.expand_dims(ms.Tensor(s_pos, ms.float32), 0))
            x_negative = net.feature_extractor(ops.expand_dims(ms.Tensor(s_neg, ms.float32), 0))
            output, _ = net.domain_classifier(x_test, x_positive, x_negative, 0, 1)
            if output == data[2][0]:
                right += 1
            total += 1
            net.set_train(True)
            train_step(ms.Tensor(data[0], ms.float32), ops.expand_dims(ms.Tensor(s_pos, ms.float32), 0), ops.expand_dims(ms.Tensor(s_neg, ms.float32), 0), ops.expand_dims(ms.Tensor(s_pos, ms.float32), 0), ms.Tensor([[output, 0, 1, 0]]), ms.Tensor([[3, 1, 1, 1]]))
            net.set_train(False)
   
    online_test(model, online_test_dataset_normal)
    online_test(model, online_test_dataset_anomaly)
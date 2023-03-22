

import os

import numpy as np
import time

import paddle
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import process_data
import argparse


def parse_parameters():
    parser = argparse.ArgumentParser(description='FalconBaselIne')
    parser.add_argument('--role', type=int, default=0, help='sum the integers (default: find the max)')

    parser.add_argument('--server', type=str, default="127.0.0.7", help='redis ip address')
    parser.add_argument('--port', type=int, default=6379, help='redis port')

    parser.add_argument('--ip_addr', type=str, default="localhost", help='my ip address')
    # train cfg
    parser.add_argument('--feature_num', type=int, default=16, help="feature number")

    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epoch_num', type=int, default=10, help="Epoch number")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--label_num', type=int, default=2, help="label number")

    # dataset path
    parser.add_argument('--mpc_data_dir', type=str, default="./mpc_data/", help="dataset path")
    parser.add_argument('--dataset_name', type=str, default="bank_market", help="dataset name")
    parser.add_argument('--output_path', type=str, default="./mpc_infer_data/", help="dataset name")

    args = parser.parse_args()
    return args


def train_valid(pargs, x, y, place, loader, test_loader):
    y_pre = pfl_mpc.layers.fc(input=x, size=1)
    cost = pfl_mpc.layers.sigmoid_cross_entropy_with_logits(y_pre, y)

    infer_program = fluid.default_main_program().clone(for_test=False)

    avg_loss = pfl_mpc.layers.mean(cost)
    optimizer = pfl_mpc.optimizer.SGD(learning_rate=pargs.lr)
    optimizer.minimize(avg_loss)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # start training
    start_time = time.time()
    step = 0
    for epoch_id in range(pargs.epoch_num):
        # feed data via loader
        for sample in loader():
            step += 1
            exe.run(feed=sample, fetch_list=[cost.name])
            batch_end = time.time()
            if step % 50 == 0:
                print(f'Training, Epoch={epoch_id}, Step={step}, Time since begin={batch_end-start_time}')

    # start prediction
    prediction_file = os.path.join(pargs.output_path, f"{pargs.dataset_name}_debug_prediction.part{pargs.role}")
    print(f"Training done, begin to do the prediction, and save to {prediction_file} ")
    for sample in test_loader():
        prediction = exe.run(program=infer_program, feed=sample, fetch_list=[cost])
        with open(prediction_file, 'ab') as f:
            f.write(np.array(prediction).tostring())


def reading_dataset(pargs, mpc_du, place):

    print(f"reading data from {pargs.mpc_data_dir}")

    x = pfl_mpc.data(name='x', shape=[pargs.batch_size, pargs.feature_num], dtype='int64')
    y = pfl_mpc.data(name='y', shape=[pargs.batch_size, 1], dtype='int64')

    # train_reader
    feature_reader = mpc_du.load_shares(
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_2_feature"),
        id=pargs.role, shape=(pargs.feature_num,))
    label_reader = mpc_du.load_shares(
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_2_label"),
        id=pargs.role, shape=(1,))

    batch_feature = mpc_du.batch(feature_reader, pargs.batch_size, drop_last=True)
    batch_label = mpc_du.batch(label_reader, pargs.batch_size, drop_last=True)

    # async data loader
    loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=pargs.batch_size)
    batch_sample = paddle.reader.compose(batch_feature, batch_label)
    loader.set_batch_generator(batch_sample, places=place)

    # test_reader
    test_feature_reader = mpc_du.load_shares(
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_2_test_feature"),
        id=pargs.role, shape=(16,))
    test_label_reader = mpc_du.load_shares(
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_2_test_label"),
        id=pargs.role, shape=(1,))

    test_batch_feature = mpc_du.batch(test_feature_reader, pargs.batch_size, drop_last=True)
    test_batch_label = mpc_du.batch(test_label_reader, pargs.batch_size, drop_last=True)

    # async data loader
    test_loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=pargs.batch_size)
    test_batch_sample = paddle.reader.compose(test_batch_feature, test_batch_label)
    test_loader.set_batch_generator(test_batch_sample, places=place)

    return x, y, loader, test_loader


def check_args():

    if args.label_num > 2:
        print("Only support 1 class")
        raise "Only support 2 class classification tasks !"

    if not os.path.exists(args.mpc_data_dir):
        raise ValueError(f"{args.mpc_data_dir} is not found. Please prepare encrypted data.")

    if not os.path.exists(args.output_path):
        raise ValueError(f"{args.output_path} is not found. Please mkdir {args.output_path}")

    train_test_data_files = [
        os.path.join(args.mpc_data_dir, f"{args.dataset_name}_2_feature.part{args.role}"),
        os.path.join(args.mpc_data_dir, f"{args.dataset_name}_2_label.part{args.role}"),
        os.path.join(args.mpc_data_dir, f"{args.dataset_name}_2_test_feature.part{args.role}"),
        os.path.join(args.mpc_data_dir, f"{args.dataset_name}_2_test_label.part{args.role}")
    ]

    for f in train_test_data_files:
        if not os.path.exists(f):
            raise ValueError(f"{f} is not found.")


if __name__ == "__main__":

    # python3 falcon/process_data.py
    # bash run_standalone_customer.sh falcon/customer_fc_sigmod.py

    args = parse_parameters()
    print(args)

    check_args()

    mpc_protocol_name = 'aby3'
    _mpc_du = get_datautils(mpc_protocol_name)
    pfl_mpc.init(mpc_protocol_name, args.role, args.ip_addr, args.server, args.port)

    # hardware
    _place = fluid.CPUPlace()

    # load data
    print("begin to load dataset .... ")
    x, y, loader, test_loader = reading_dataset(args, _mpc_du, _place)
    # trained prediction
    print("begin to train and predict .... ")
    train_valid(args, x, y, _place, loader, test_loader)

    print("Prediction done, jointly to decrypt \n")
    process_data.decrypt_data_to_file(
        os.path.join(args.output_path, f"{args.dataset_name}_debug_prediction"),
        (args.batch_size,),
        os.path.join(args.output_path, f"{args.dataset_name}_output_prediction"))



import argparse
import os


def parse_parameters():
    parser = argparse.ArgumentParser(description='FalconBaselIne')

    parser.add_argument('--class_num', type=int, default=2, help="label number")

    parser.add_argument('--input_dataset', type=str, default="./falcon/bank_marketing_data/bank.data.norm",
                        help="dataset path")
    parser.add_argument('--dataset_name', type=str, default="bank_market", help="dataset path")
    parser.add_argument('--mpc_data_dir', type=str, default="./mpc_data/", help="dataset path")

    args = parser.parse_args()
    return args


def generate_encrypted_train_data(pargs, train_dataset, mpc_du):
    """
    generate encrypted samples
    """

    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in train_dataset:
            yield mpc_du.make_shares(np.array(instance[0]))

    def encrypted_mnist_labels():
        """
        label reader
        """
        for instance in train_dataset:
            if pargs.class_num == 2:
                label = np.array(1) if instance[1] == 1 else np.array(0)
            elif pargs.class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(pargs.class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(
        encrypted_mnist_features,
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_{pargs.class_num}_feature"))
    mpc_du.save_shares(
        encrypted_mnist_labels,
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_{pargs.class_num}_label"))


def generate_encrypted_test_data(pargs, test_dataset, mpc_du):
    """
    generate encrypted samples
    """

    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in test_dataset:
            yield mpc_du.make_shares(np.array(instance[0]))

    def encrypted_mnist_labels():
        """
        label reader
        """
        test_label_file_name = os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_test_label")
        for instance in test_dataset:
            if pargs.class_num == 2:
                label = np.array(1) if instance[1] == 0 else np.array(0)
                with open(test_label_file_name, 'a+') as f:
                    f.write(str(1 if instance[1] == 0 else 0) + '\n')
            elif pargs.class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
                with open(test_label_file_name, 'a+') as f:
                    f.write(str(instance[1]) + '\n')
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(pargs.class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(
        encrypted_mnist_features,
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_{pargs.class_num}_test_feature"))
    mpc_du.save_shares(
        encrypted_mnist_labels,
        os.path.join(pargs.mpc_data_dir, f"{pargs.dataset_name}_{pargs.class_num}_test_label"))


def read_a_file(filename):
    result = list()
    with open(filename) as file:
        for line in file:
            line_list = line.strip().split(",")
            result.append(([float(ele) for ele in line_list[:-1]], int(float(line_list[-1])) ))
    print(f"loading data done, feature dim = {len(result[0][0])}, label_size = {result[0][1]}")
    return result


def split_train_test(result: list, RATE):
    total_size = len(result)
    train_index = RATE * total_size

    train_data = []
    test_data = []

    for index, ele in enumerate(result):
        if index <train_index:
            train_data.append(ele)
        else:
            test_data.append(ele)
    return train_data, test_data


def check_args():

    if args.class_num > 2:
        print("Only support 1 class")
        raise "Only support 2 class classification tasks !"

    if not os.path.exists(args.mpc_data_dir):
        raise ValueError(f"{args.mpc_data_dir} is not found. Please prepare encrypted data.")

    if not os.path.exists(args.input_dataset):
        raise ValueError(f"{args.input_dataset} is not found. Please mkdir {args.input_dataset}")


if __name__ == "__main__":

    # python3 falcon/process_data.py

    args = parse_parameters()
    print(args)

    check_args()

    import os
    import logging

    print(f"1. reading dataset from {args.input_dataset}")
    all_data = read_a_file(args.input_dataset)
    train_dataset, test_dataset = split_train_test(all_data, 0.8)
    print(f"2. data partitioned, train:test = {len(train_dataset)/len(all_data)}: {len(test_dataset)/len(all_data)}")

    import numpy as np
    from paddle_fl.mpc.data_utils.data_utils import get_datautils

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("fluid")
    logger.setLevel(logging.INFO)

    mpc_du = get_datautils('aby3')

    print("3. begin to convert both train and test datasets")
    generate_encrypted_train_data(args, train_dataset, mpc_du)
    generate_encrypted_test_data(args, test_dataset, mpc_du)

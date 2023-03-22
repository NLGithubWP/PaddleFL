# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Process data for MNIST: 10 classes.
"""


def generate_encrypted_train_data(mpc_data_dir, class_num):
    """
    generate encrypted samples
    """

    def encrypted_mnist_features():
        """
        feature reader
        """
        if_print = 0
        for instance in train_dataset:
            if if_print == 0:
                # this is 784 features, 5
                print(f"data = {instance}, \n {len(instance[0])}, \n {instance[1]}")
                if_print= 1
            yield mpc_du.make_shares(np.array(instance[0]))

    def encrypted_mnist_labels():
        """
        label reader
        """
        if_print = 0
        for instance in train_dataset:
            if class_num == 2:
                label = np.array(1) if instance[1] == 1 else np.array(0)
                if if_print == 0:
                    print(f" ins_1 is  {instance[1]}, label is  {label}")
                    if_print = 1
            elif class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(encrypted_mnist_features, mpc_data_dir + "bank_market{}_feature".format(class_num))
    mpc_du.save_shares(encrypted_mnist_labels, mpc_data_dir + "bank_market{}_label".format(class_num))


def generate_encrypted_test_data(mpc_data_dir, class_num, label_mnist_filepath):
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
        for instance in test_dataset:
            if class_num == 2:
                label = np.array(1) if instance[1] == 0 else np.array(0)
                with open(label_mnist_filepath, 'a+') as f:
                    f.write(str(1 if instance[1] == 0 else 0) + '\n')
            elif class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
                with open(label_mnist_filepath, 'a+') as f:
                    f.write(str(instance[1]) + '\n')
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(encrypted_mnist_features, mpc_data_dir + "bank_market{}_test_feature".format(class_num))
    mpc_du.save_shares(encrypted_mnist_labels, mpc_data_dir + "bank_market{}_test_label".format(class_num))


def load_decrypt_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = mpc_du.reconstruct(np.array(instance))
        logger.info(p)


def load_decrypt_bs_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = np.bitwise_xor(np.array(instance[0]), np.array(instance[1]))
        p = np.bitwise_xor(p, np.array(instance[2]))
        logger.info(p)


def decrypt_data_to_file(filepath, shape, decrypted_filepath):
    """
    load the encrypted data (arithmetic share) and reconstruct to a file
    """
    import six
    import os
    import numpy as np
    import six
    import paddle
    from paddle_fl.mpc.data_utils.data_utils import get_datautils
    mpc_du = get_datautils('aby3')

    if os.path.exists(decrypted_filepath):
        os.remove(decrypted_filepath)
    part_readers = []
    print("1. adding share to part readers")
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    print("2. write to file")
    for instance in mpc_share_reader():
        p = mpc_du.reconstruct(np.array(instance))
        with open(decrypted_filepath, 'a+') as f:
            for i in p:
                if i > 0.5:
                    f.write(str(1) + '\n')
                else :
                    f.write(str(0) + '\n')


def decrypt_bs_data_to_file(filepath, shape, decrypted_filepath):
    """
    load the encrypted data (boolean share) and reconstruct to a file
    """
    if os.path.exists(decrypted_filepath):
        os.remove(decrypted_filepath)
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = np.bitwise_xor(np.array(instance[0]), np.array(instance[1]))
        p = np.bitwise_xor(p, np.array(instance[2]))
        with open(decrypted_filepath, 'a+') as f:
            for i in p:
                f.write(str(i) + '\n')


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


if __name__ == '__main__':

    import os
    import time
    import logging

    mpc_data_dir = './mpc_data/'
    label_mnist_filepath = mpc_data_dir + "test_label_bank_market"
    if not os.path.exists(mpc_data_dir):
        os.mkdir(mpc_data_dir)
    else:
        for exit_file in os.listdir(mpc_data_dir):
            os.remove(os.path.join(mpc_data_dir, exit_file ))

    if os.path.exists(label_mnist_filepath):
        os.remove(label_mnist_filepath)

    all_data = read_a_file("./falcon/bank_marketing_data/bank.data.norm")
    train_dataset, test_dataset = split_train_test(all_data, 0.8)
    print(f"data partitioned, train:test = {len(train_dataset)/len(all_data)}: {len(test_dataset)/len(all_data)}")


    import numpy as np
    import six
    import paddle
    from paddle_fl.mpc.data_utils.data_utils import get_datautils

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("fluid")
    logger.setLevel(logging.INFO)

    mpc_du = get_datautils('aby3')
    # sample_reader = paddle.dataset.mnist.train()
    # test_reader = paddle.dataset.mnist.test()

    class_num = 2
    generate_encrypted_train_data(mpc_data_dir, class_num)
    generate_encrypted_test_data(mpc_data_dir, class_num, label_mnist_filepath)

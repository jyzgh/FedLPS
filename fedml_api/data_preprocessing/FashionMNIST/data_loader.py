import json
import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from .datasets import FashionMNIST_truncated


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_FashionMNIST_by_device_id(batch_size,
                                           device_id,
                                           train_path="FashionMNIST_mobile",
                                           test_path="FashionMNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_FashionMNIST(batch_size, train_path, test_path)


def load_partition_data_FashionMNIST(batch_size,
                              train_path="./../../../data/FashionMNIST/train",
                              test_path="./../../../data/FashionMNIST/test"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_data_FashionMNIST_custom(dataset, data_dir, partition_method, partition_alpha, client_num_in_total,
                                     client_number, batch_size,
                                     train_path="./../../../data/FashionMNIST/train",
                                     test_path="./../../../data/FashionMNIST/test"):
    # users, groups, train_data, test_data = read_data(train_path, test_path)
    #
    # if len(groups) == 0:
    #     groups = [None for _ in users]
    # train_data_num = 0
    # test_data_num = 0
    # train_data_local_dict = dict()
    # test_data_local_dict = dict()
    # train_data_local_num_dict = dict()
    # train_data_global = list()
    # test_data_global = list()
    # client_idx = 0
    # logging.info("loading data...")
    # for u, g in zip(users, groups):
    #     user_train_data_num = len(train_data[u]['x'])
    #     user_test_data_num = len(test_data[u]['x'])
    #     train_data_num += user_train_data_num
    #     test_data_num += user_test_data_num
    #     train_data_local_num_dict[client_idx] = user_train_data_num
    #
    #     # transform to batches
    #     train_batch = batch_data(train_data[u], batch_size)
    #     test_batch = batch_data(test_data[u], batch_size)
    #
    #     # index using client index
    #     train_data_local_dict[client_idx] = train_batch
    #     test_data_local_dict[client_idx] = test_batch
    #     train_data_global += train_batch
    #     test_data_global += test_batch
    #     client_idx += 1
    # logging.info("finished the loading data")
    # client_num = client_idx
    # class_num = 10





    # ===========================================================================================================

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local


    logging.info("finished the customized partial data")

    # return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    #        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_data_FashionMNIST_fast(dataset, data_dir, partition_method, partition_alpha, client_num_in_total,
                                   client_number, batch_size,
                                   train_path="./../../../data/FashionMNIST/train",
                                   test_path="./../../../data/FashionMNIST/test"):
    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")

    all_train_data_dict = dict()
    all_train_data_dict['x'] = []
    all_train_data_dict['y'] = []
    all_test_data_dict = dict()
    all_test_data_dict['x'] = []
    all_test_data_dict['y'] = []
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        all_train_data_dict['x'].extend(train_data[u]['x'])
        all_train_data_dict['y'].extend(train_data[u]['y'])
        all_test_data_dict['x'].extend(test_data[u]['x'])
        all_test_data_dict['y'].extend(test_data[u]['y'])
    all_train_data_dict['y'] = list(map(int, all_train_data_dict['y']))
    all_test_data_dict['y'] = list(map(int, all_test_data_dict['y']))
    x_train_arr = np.array(all_train_data_dict['x'])
    y_train_arr = np.array(all_train_data_dict['y'])

    logging.info("*********partition data***************")
    n_train = len(all_test_data_dict['x'])
    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition_method == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, client_number)
        net_dataidx_map = {i: batch_idxs[i] for i in range(client_number)}

    elif partition_method == "hetero":
        min_size = 0
        K = 10
        # N = y_train.shape[0]
        # y_train_arr = np.array(all_train_data_dict['y'])
        N = len(y_train_arr)
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 5:
            idx_batch = [[] for _ in range(client_number)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train_arr == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(partition_alpha, client_number))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / client_number) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                # min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(len(idx_batch)):
                if len(idx_batch[j]) < 5:
                    for v in range(len(idx_batch)):
                        if len(idx_batch[v]) > 10:
                            idx_batch[j] = idx_batch[v][:-5]
                            del idx_batch[v][:-5]
                            break
            min_size = min([len(idx_j) for idx_j in idx_batch])
            assert min_size >= 5

        for j in range(client_number):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # elif partition == "hetero-fix":
    #     dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
    #     net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition_method == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        # traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train_arr, net_dataidx_map)

    # -----------------------------partition finished-----------------------------

    class_num = len(np.unique(all_train_data_dict['y']))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    logging.info("train_dl_global number = " + str(len(all_train_data_dict['x'])))
    logging.info("test_dl_global number = " + str(len(all_test_data_dict['x'])))
    test_data_num = len(all_test_data_dict['x'])

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        # train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
        #                                                    dataidxs)

        train_data_local_dict['x'] = x_train_arr[dataidxs]
        train_data_local_dict['y'] = y_train_arr[dataidxs]
        test_data_local_dict = all_test_data_dict

        # transform to batches
        train_data_local = batch_data(train_data_local_dict, batch_size)
        test_data_local = batch_data(test_data_local_dict, batch_size)

        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    logging.info("finished the customized partial data")

    # return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
    #        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_FashionMNIST_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # elif partition == "hetero-fix":
    #     dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
    #     net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        # traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def load_FashionMNIST_data(datadir):
    train_transform, test_transform = _data_transforms_FashionMNIST()

    FashionMNIST_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    FashionMNIST_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)
    # FashionMNIST_train_ds = FashionMNIST_truncated(datadir, train=True, download=True)
    # FashionMNIST_test_ds = FashionMNIST_truncated(datadir, train=False, download=True)

    X_train, y_train = FashionMNIST_train_ds.data, FashionMNIST_train_ds.target
    X_test, y_test = FashionMNIST_test_ds.data, FashionMNIST_test_ds.target

    return (X_train, y_train, X_test, y_test)


def _data_transforms_FashionMNIST():
    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.352,))
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.352,)),
    ])

    return train_transform, valid_transform


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    # add num_workers and pin_memory.
    # train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=False)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=False)

    return train_dl, test_dl


def get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl

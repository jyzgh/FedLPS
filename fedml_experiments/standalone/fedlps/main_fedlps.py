import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchstat import stat
import wandb



sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.SVHN.data_loader import load_partition_data_SVHN
from fedml_api.model.cv.cnn import CNN_OriginalFedAvg
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist, load_partition_data_mnist_custom,\
    load_partition_data_mnist_fast
from fedml_api.data_preprocessing.FashionMNIST.data_loader import load_partition_data_FashionMNIST_custom
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.standalone.fedavg.fedlps_api import FedLPSAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("-pf", "--prefix", type=str, default='tmp', metavar='PFX',
                        help='dataset prefix for logging & checkpoint saving')

    parser.add_argument('-net', '--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('-ds', '--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers (\'hetero\' or \'homo\')')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_schedule', type=int, default=0,
                        help='Adjusting learning rate or not (default: 0)')

    parser.add_argument('--lr_gama', type=int, default=0.2,
                        help='gama in learning rate scheduler (default: 0.2)')

    parser.add_argument('--milestones', type=list, default=[25, 50, 75],
                        help='Adjusting learning rate (default: [25, 50, 75])')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('-cr', '--comm_round', type=int, default=100,
                        help='how many round of communications we should use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('-dp', '--dataparallel', type=int, default=0,
                        help='dataparallel (default: 0)')

    parser.add_argument('--pr_type', type=str, default=0,
                        help='Type of FL.')

    parser.add_argument('--pr_round', type=int, default=0,
                        help='dataparallel (default: 0)')

    parser.add_argument('--pr_strategy', type=str, default='0',
                        help='strategy for pruning model (default: \'0\', i.e. no pruning.)')

    parser.add_argument('--order_pr', type=int, default='1',
                        help='Ordered aggregate models (default: 0)')

    parser.add_argument('-fp', '--freeze_pruning', type=int, default=0,
                        help='Prune frozen layers or not. (1: Prune frozen layers; 0: Not prune frozen layers)')

    parser.add_argument('-lt', '--local_test', type=int, default=0,
                        help='Test on local model or global model. (1: local model; 0: global model)')

    parser.add_argument('-fl', '--freeze_layer', type=int, default=0,
                        help='Freeze some layers for sharing parameters.')

    parser.add_argument('--pareto', type=int, default=0,
                        help='Enable pareto-optimization')

    parser.add_argument('-uc', '--update_client', type=int, default=1,
                        help='Sample clients before each local training')

    parser.add_argument('-fm', '--feature_map', type=bool, default=False,
                        help='make feature maps')

    parser.add_argument('-mu', '--mu', type=float, default=0.2,
                        help='Parameter \'mu\' for fedprox.')

    parser.add_argument('-seed', '--seed', type=int, default=1111,
                        help='Random seed.')

    parser.add_argument('--resume', '-rs', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--store_ckpt', '-ckpt', default=0, type=int,
                        help='Store the checkpoint or not (default: 0)')

    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name in ["mnist", "mn"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.data_dir = "./../../../data/MNIST"

        if "cnn" in args.model:
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_mnist_fast(args.dataset, args.data_dir, args.partition_method,
                                                       args.partition_alpha, args.client_num_in_total,
                                                       args.client_num_in_total, args.batch_size)
        else:
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = load_partition_data_mnist_custom(args.dataset, args.data_dir, args.partition_method,
                                                       args.partition_alpha, args.client_num_in_total,
                                                       args.client_num_in_total, args.batch_size)
    elif dataset_name in ["fashionmnist", "FashionMNIST", "fmn", "fmnist"]:
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.data_dir = "./../../../data/FashionMNIST"

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_FashionMNIST_custom(args.dataset, args.data_dir, args.partition_method,
                                                     args.partition_alpha, args.client_num_in_total,
                                                     args.client_num_in_total, args.batch_size)

        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """

    else:
        if dataset_name in ["cifar10", "cf10"]:
            data_loader = load_partition_data_cifar10
        elif dataset_name in ["cifar100", "cf100"]:
            data_loader = load_partition_data_cifar100
        elif dataset_name in ["svhn", "SVHN", "sn"]:
            logging.info("load_data. dataset_name = %s" % dataset_name)
            args.data_dir = "./../../../data/SVHN"
            data_loader = load_partition_data_SVHN
        else:
            data_loader = load_partition_data_cifar10
            logging.warning("Unknown dataset! (dataset_name = %s)" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "cnn":
        if args.dataset == "femnist":
            model = CNN_OriginalFedAvg(False)
        else:
            model = CNN_OriginalFedAvg()

    elif model_name in ["pre_r18", "pre_resnet18"]:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, output_dim)
    elif model_name in ["r18", "resnet18"]:
        model = torchvision.models.resnet18(num_classes=output_dim)

    elif model_name in ["pre_SqueezeNet", "pre_squeezenet", "pre_sqnet"]:
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, output_dim, kernel_size=1)
    elif model_name in ["SqueezeNet", "squeezenet", "sqnet"]:
        model = torchvision.models.squeezenet1_1(num_classes=output_dim)
        # model.classifier[0] = nn.Sequential()

    elif model_name in ["pre_ShuffleNet", "pre_shufflenet", "pre_sfnet"]:
        if args.resume:
            model = load_resume(args)
        if model is not None:
            model.fc = nn.Linear(1024, output_dim)
        else:
            model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
            model.fc = nn.Linear(1024, output_dim)
    elif model_name in ["ShuffleNet", "shufflenet", "sfnet"]:
        model = torchvision.models.shufflenet_v2_x1_0(num_classes=output_dim)
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)

def load_resume(args):
    if os.path.isfile(args.resume):
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # Try to load the whole model.
        try:
            model = checkpoint['model']
            try:
                round = args.comm_round - checkpoint['round']
            except (KeyError, TypeError):
                logging.info('\'round\' of the checkpoint not found.')
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
        except (KeyError, TypeError):
            logging.info('\'model\' of the checkpoint not found.')
            # The whole model is not saved, try to load weights.
            try:
                round = args.comm_round - checkpoint['round']
            except (KeyError, TypeError):
                logging.info('\'round\' of the checkpoint not found.')
            try:
                best_acc = checkpoint['best_acc']
            except (KeyError, TypeError):
                logging.info('\'best_acc\' of the checkpoint not found.')
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except (KeyError, TypeError, RuntimeError, UnboundLocalError):
                # Model saved as 'torch.save(model, path)'.
                logging.info('\'state_dict\' of the checkpoint not found.')
                model = torch.load(args.resume)

        logging.info("=> loaded checkpoint '{}' (round {})"
                     .format(args.resume, round))
        return model
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
        return None

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    if args.dataparallel == 1:
        device = torch.device("cuda:{}".format(torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
        # logger.info(torch.cuda.device_count())
        logger.info(device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        logger.info(device)
    # device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="fedml",
        # name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        name=str(args.prefix) + "-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU
    torch.cuda.manual_seed_all(args.seed)  # All GPUs
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    if args.pr_type in ["fedlps"]:
        fedavgAPI = FedLPSAPI(dataset, device, args, model_trainer)
        fedavgAPI.train()
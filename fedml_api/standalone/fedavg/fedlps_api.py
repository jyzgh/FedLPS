import copy
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import torchvision
from torchvision import transforms
import wandb
import matplotlib.pyplot as plt
# from thop import profile
# from torchstat import stat
from fedml_api.standalone.fedavg.client import Client


class FedLPSAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

        self.prune_strategy = args.pr_strategy
        self._ln_prune = self._ln_prune_r18
        self.prune_prob = {
            '0': [0, 0, 0, 0],
            'AD': [0, 0, 0, 0],
            '0.1': [0.1, 0.1, 0.1, 0.1],
            '0.2': [0.2, 0.2, 0.2, 0.2],
            '0.3': [0.3, 0.3, 0.3, 0.3],
            '0.4': [0.4, 0.4, 0.4, 0.4],
            '0.5': [0.5, 0.5, 0.5, 0.5],
            '0.6': [0.6, 0.6, 0.6, 0.6],
            '0.7': [0.7, 0.7, 0.7, 0.7],
            '0.8': [0.8, 0.8, 0.8, 0.8],
            '0.9': [0.9, 0.9, 0.9, 0.9],
        }
        # self.ad_prob = ['0', '0.1', '0.3', '0.5', '0.7']
        self.ad_prob = ['0', '0.2', '0.4', '0.6', '0.8']
        self.ad_prob_same = ['0', '0.2', '0.4', '0.6', '0.8']
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("=================%s training on %s================="
                    % (self.args.pr_type, self.args.dataset))
        w_global = copy.deepcopy(self.model_trainer.get_model_params())
        init_model = copy.deepcopy(self.model_trainer.get_model())

        # **************Test***************
        # test_model = torchvision.models.resnet18()
        # w_init_model = init_model.state_dict()
        # w_test_model = test_model.state_dict()
        # print(init_model)
        # print(test_model)
        # stat(init_model, (3, 28, 28))
        # print(list(init_model.named_buffers()))
        # **************Test***************

        # modify adaptive pruning ratios and freeze layers.
        # --------ResNet--------
        if self.args.model in ["pre_r18", "pre_resnet18", "r18", "resnet18"]:
            # freeze layers
            init_model = self._freeze_layers_r18(self.args.freeze_layer, init_model)
            # modify adaptive pruning ratios
            if self.args.pr_type == "fedlps":
                if self.args.freeze_layer == 1:
                    self.ad_prob = ['0', '0', '0.2', '0.5', '0.8']
                if self.args.freeze_layer == 2:
                    self.ad_prob = ['0', '0', '0', '0.4', '0.8']
            elif self.args.pr_type == "hermes":
                self.ad_prob = ['0', '0.2', '0.4', '0.6', '0.8']
            self._ln_prune = self._ln_prune_r18
        # --------SqueezeNet--------
        elif self.args.model in ["pre_SqueezeNet", "pre_squeezenet", "pre_sqnet", "SqueezeNet", "squeezenet", "sqnet"]:
            # freeze layers
            init_model = self._freeze_layers_sqnet(self.args.freeze_layer, init_model)
            # modify adaptive pruning ratios
            if self.args.pr_type == "fedlps":
                if self.args.freeze_layer == 1:  # freeze the 1st Fire block (features.3).
                    self.ad_prob = ['0', '0', '0.1', '0.3', '0.4']
                if self.args.freeze_layer == 2:  # freeze 3 Fire blocks (features.6).
                    self.ad_prob = ['0', '0', '0', '0.2', '0.4']
            elif self.args.pr_type == "hermes":
                self.ad_prob = ['0', '0.1', '0.2', '0.3', '0.4']
            self._ln_prune = self._ln_prune_sqnet
        # --------MobileNet--------
        elif self.args.model in ["pre_mobilenet", "pre_mbnet", "mobilenet", "mbnet"]:
            # freeze layers
            init_model = self._freeze_layers_mbnet(self.args.freeze_layer, init_model)
            # modify adaptive pruning ratios
            if self.args.pr_type == "fedlps":
                if self.args.freeze_layer == 1:  # freeze the 1st Fire block (features.3).
                    self.ad_prob = ['0', '0', '0.2', '0.5', '0.8']
                if self.args.freeze_layer == 2:  # freeze 3 Fire blocks (features.6).
                    self.ad_prob = ['0', '0', '0', '0.4', '0.8']
            elif self.args.pr_type == "hermes":
                self.ad_prob = ['0', '0.2', '0.4', '0.6', '0.8']
            self._ln_prune = self._ln_prune_mbnet
        # --------ShuffleNet--------
        elif self.args.model in ["pre_ShuffleNet", "pre_shufflenet", "pre_sfnet", "ShuffleNet", "shufflenet", "sfnet"]:
            # freeze layers
            init_model = self._freeze_layers_sfnet(self.args.freeze_layer, init_model)
            # modify adaptive pruning ratios
            if self.args.pr_type == "fedlps":
                if self.args.freeze_layer == 1:  # freeze the 1st Fire block (features.3).
                    self.ad_prob = ['0', '0', '0.2', '0.5', '0.8']
                if self.args.freeze_layer == 2:  # freeze 3 Fire blocks (features.6).
                    self.ad_prob = ['0', '0', '0', '0.4', '0.8']
            elif self.args.pr_type == "hermes":
                self.ad_prob = ['0', '0.2', '0.4', '0.6', '0.8']
            self._ln_prune = self._ln_prune_sfnet

        best_acc = 0.0
        models_prune = []  # Store the pruned models for each client.
        is_pruned = []  # Indicate whether the model of each client is pruned.

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            if self.args.update_client or round_idx == 0:
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            # Adjusting lr.
            if self.args.lr_schedule and round_idx in self.args.milestones:
                self.args.lr = self.args.lr * self.args.lr_gama

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # ---------------------------------FedLPS pruning and training------------------------------------------
                # Get local model.
                if idx >= len(models_prune):
                    pr_model = copy.deepcopy(init_model)
                    pr_model.load_state_dict(copy.deepcopy(w_global))
                else:
                    pr_model = models_prune[idx]
                    pr_model.load_state_dict(copy.deepcopy(w_global))

                # **************Test***************
                # best_acc = self._local_test_on_all_clients(round_idx, models_prune, w_global, best_acc)
                # **************Test***************

                # Local training.
                self.model_trainer.set_model(pr_model)
                w = client.train(copy.deepcopy(pr_model.state_dict()))
                pr_model.load_state_dict(w)

                # Adaptive pruning.
                if round_idx == self.args.pr_round:
                    if self.prune_strategy == "0":
                        is_pruned.append(1)
                    else:
                        if self.prune_strategy == "AD":
                            pr_strategy = self.ad_prob[idx % len(self.ad_prob)]  # get pruning ratio of specific client
                            pr_prob = self.prune_prob[pr_strategy]  # get pruning ratios for layers
                            self.prune_prob['AD'] = self.prune_prob[pr_strategy]  # copy pruning ratios for layers
                        else:
                            pr_prob = self.prune_prob[self.prune_strategy]  # get pruning ratios for layers
                        pr_model = self._ln_prune(pr_model, pr_prob, remove=0)
                        w = pr_model.state_dict()
                        logging.info("L1 pruning on client %s, prune strategy %s: %s" %
                                     (str(client_idx), self.prune_strategy, pr_prob))
                        is_pruned.append(1)

                # store pruned models
                w_locals.append((client.get_sample_number(), w))
                if idx >= len(models_prune):
                    models_prune.append(pr_model)
                else:
                    models_prune[idx].load_state_dict(w)

            # Aggregation.
            w_global = self._aggregate(w_locals)

            # ------------------------------------ Test results ----------------------------------------
            if round_idx == self.args.comm_round - 1:
                best_acc = self._local_test_on_all_clients(round_idx, models_prune, w_global, best_acc)
                # best_acc = self._local_test_on_all_clients(round_idx, init_model, is_pruned, best_acc)

            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                best_acc = self._local_test_on_all_clients(round_idx, models_prune, w_global, best_acc)
                # best_acc = self._local_test_on_all_clients(round_idx, init_model, is_pruned, best_acc)

    def _freeze_layers_sfnet(self, layer_num, model):
        if layer_num == 0:
            return model
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding stages: {0, 2, 3, 4, conv5.0} (not break InvertedResidual block)
        frozen_modules = [0, 55, 149, 199, 203] # index: layer_num; value: frozen modules until this layer_num.
        for idx, m in enumerate(model.named_modules()):
            if idx <= 1: # 0 -- squeezenet, 1 -- conv1(this is a nn.Sequential, not a conv layer)
                continue
            if idx <= frozen_modules[layer_num]:
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    m[1].requires_grad_(False)
        return model

    def _freeze_layers_mbnet(self, layer_num, model):
        if layer_num == 0:
            return model
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding features: {0, 6.0.2, 10.2.relu, avgpool, classifier.2} (break InvertedResidual block)
        # frozen_modules = [0, 89, 168, 196, 200]  # index: layer_num; value: frozen modules until this layer_num.
        #corresponding features: {0, 5, 9, avgpool, classifier.2} (not break InvertedResidual block)
        frozen_modules = [0, 83, 155, 196, 200] # index: layer_num; value: frozen modules until this layer_num.
        for idx, m in enumerate(model.named_modules()):
            if idx <= 1: # 0 -- squeezenet, 1 -- features
                continue
            if idx <= frozen_modules[layer_num]:
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    m[1].requires_grad_(False)
        return model

    def _freeze_layers_sqnet(self, layer_num, model):
        if layer_num == 0:
            return model
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding fire block index: {0, 1, 3, 4, 8}, features: {0, 3, 6, 7, 12}
        frozen_modules = [0, 11, 26, 33, 62] # index: layer_num; value: frozen modules until this layer_num.
        idx = 0
        for name, m in enumerate(model.named_modules()):
            if idx <= 1: # 0 -- squeezenet, 1 -- features
                idx += 1
                continue
            if idx <= frozen_modules[layer_num]:
                m[1].requires_grad_(False)
            idx += 1
        return model

    def _freeze_layers_r18(self, layer_num, model):
        if layer_num == 0:
            return model
        if layer_num == 1:
            for name, child in model.named_children():
                if "layer2" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 2:
            for name, child in model.named_children():
                if "layer3" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 3:
            for name, child in model.named_children():
                if "layer4" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 4:
            for name, child in model.named_children():
                if "avgpool" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        return model

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _ln_prune_sfnet(self, glb_model, pr_prob, remove=0):
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding stages: {0, 2, 3, 4, conv5.0} (not break stage block)
        frozen_modules = [0, 55, 149, 199, 203] # index: layer_num; value: frozen modules until this layer_num.
        prune_count = 0

        for idx, m in enumerate(glb_model.named_modules()):
            if not self.args.freeze_pruning and frozen_modules[self.args.freeze_layer] >= idx:
                # Not prune frozen layers.
                continue

            if isinstance(m[1], nn.Conv2d):
                if "conv1" in m[0]:
                    # The first conv layer in sfnet.
                    continue
                if m[0] == "stage2.0.branch1.0" or prune_count == 0:  # stage2's 1st conv layer.
                    # Pruning 'out_planes'.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                else:  # other conv layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                prune_count += 1

            elif isinstance(m[1], nn.BatchNorm2d):
                if "conv1" in m[0]:
                    # The first BN layer in sfnet.
                    continue
                torch_prune.l1_unstructured(m[1], name="weight", amount=pr_prob[0])

            elif isinstance(m[1], nn.Linear):
                if prune_count == 0:  # 1st layer in un-pruned sub-model.
                    if m[0] == "fc":  # The 1st un-pruned layer is the last fc layer.
                        break
                    else:
                        torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                        prune_count += 1
                        continue
                elif m[0] == "fc":  # last fc layer.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    prune_count += 1
                    break
                else:  # other fc layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                    prune_count += 1

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        # dummy_input = torch.randn(1, 3, 32, 32)  # .to(device)
        # flops, params = profile(glb_model, (dummy_input,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        return glb_model

    def _ln_prune_mbnet(self, glb_model, pr_prob, remove=0):
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding features: {0, 6.0.2, 10.2.relu, avgpool, classifier.2} (break InvertedResidual block)
        # frozen_modules = [0, 89, 168, 196, 200]  # index: layer_num; value: frozen modules until this layer_num.
        #corresponding features: {0, 5, 9, avgpool, classifier.2} (not break InvertedResidual block)
        frozen_modules = [0, 83, 155, 196, 200] # index: layer_num; value: frozen modules until this layer_num.
        prune_count = 0

        for idx, m in enumerate(glb_model.named_modules()):
            if not self.args.freeze_pruning and frozen_modules[self.args.freeze_layer] >= idx:
                # Not prune frozen layers.
                continue
            if m[0] == "features.0.0":
                # The first conv layer in mbnet.
                continue

            if isinstance(m[1], nn.Conv2d):
                if m[0] == "features.1.block.0.0" or prune_count == 0:  # InvertedResidual_block1's 1st conv layer.
                    # Pruning 'out_planes'.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                else:  # other conv layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                prune_count += 1

            elif isinstance(m[1], nn.BatchNorm2d):
                torch_prune.l1_unstructured(m[1], name="weight", amount=pr_prob[0])

            elif isinstance(m[1], nn.Linear):
                if prune_count == 0:  # 1st layer in un-pruned sub-model.
                    if m[0] == "classifier.3":  # The 1st un-pruned layer is the last fc layer.
                        break
                    else:
                        torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                        prune_count += 1
                        continue
                elif m[0] == "classifier.3":  # last fc layer.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    prune_count += 1
                    break
                else:  # other fc layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                    prune_count += 1

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        # dummy_input = torch.randn(1, 3, 32, 32)  # .to(device)
        # flops, params = profile(glb_model, (dummy_input,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        return glb_model

    def _ln_prune_sqnet(self, glb_model, pr_prob, remove=0):
        # args.freeze_layer: {0, 1, 2, 3, 4}
        # corresponding fire blocks: {0, 1, 3, 4, 8}, features: {0, 3, 6, 7, 12}, frozen_modules: {0, 11, 26, 33, 62}
        frozen_conv = [0, 3, 9, 12, 24]  # index -- freeze_layer; value -- frozen conv until this freeze_layer.
        conv_count = 0  # Total 26 conv layers in squeezenet. The last one is for prediction.
        prune_count = 0  # Number of pruned layers.

        for name, module in glb_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in squeezenet.
                    conv_count += 1
                    continue

                # Not prune frozen layers.
                if not self.args.freeze_pruning and frozen_conv[self.args.freeze_layer] >= conv_count:
                    conv_count += 1
                    continue

                else:  # Normal conv layers in blocks.
                    if conv_count == 1 or prune_count == 0:  # 1st conv layer in fire_block1 or unfrozen sub-model.
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                        prune_count += 1
                        conv_count += 1
                        continue
                    elif conv_count < 25:  # other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                        prune_count += 1
                        conv_count += 1
                        continue
                    elif conv_count == 25:  # the last conv layer is the prediction layer.
                        # Pruning 'in_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        prune_count += 1
                        conv_count += 1
                        continue
                    else:
                        conv_count += 1
                        continue

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        return glb_model

    def _ln_prune_r18(self, glb_model, pr_prob, remove=0):
        # index: frozen stage; value: conv number until this stage.
        # Note: values in frozen_conv is 1 bigger than it in sqnet, mbnet, and sfnet.
        frozen_conv = [0, 5, 10, 15, 20]

        conv_count = 0
        down_count = 1  # r18's stage1 has no 'downsample' layer
        for name, module in glb_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in resnet.
                    conv_count += 1
                    continue

                if 'downsample' in name:
                    # The first downsample conv layer, only prune 'out_planes'.
                    if down_count == 0:
                        # Use the pruning probability in stage1(pr_prob[0]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    else:  # The other downsample conv layer.
                        if down_count == 1:  # r18's stage1 has no 'downsample' layer
                            conv_count += 1
                            down_count += 1
                            continue
                        if not self.args.freeze_pruning and self.args.freeze_layer >= down_count+1:  # Not prune frozen layers.
                            down_count += 1
                            conv_count += 1
                            continue
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count-1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    conv_count += 1
                    continue

                else:  # Normal conv layers in blocks.
                    if not self.args.freeze_pruning and frozen_conv[self.args.freeze_layer] > conv_count:  # Not prune frozen Stage1.
                        conv_count += 1
                        continue
                    if conv_count == 1:  # Stage1's 1st conv layer.
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                    elif conv_count <= 5:  # Stage1's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)

                    elif conv_count == 6:  # Stage2's 1st conv layer.
                        if self.args.freeze_layer != 1:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)
                    elif conv_count <= 10:  # Stage2's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)

                    elif conv_count == 11:  # Stage3's 1st conv layer.
                        if self.args.freeze_layer != 2:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)
                    elif conv_count <= 15:  # Stage3's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)

                    elif conv_count == 16:  # Stage4's 1st conv layer.
                        if self.args.freeze_layer != 3:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=0)
                    elif conv_count <= 20:  # Stage4's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=0)

                    conv_count += 1
                    continue

            elif isinstance(module, nn.BatchNorm2d):
                # 'conv_count' in nn.BatchNorm2d is 1 bigger than nn.Conv2d.
                if conv_count == 1:  # The 1st bn in resnet.
                    continue

                if not self.args.freeze_pruning and frozen_conv[self.args.freeze_layer] + 1 >= conv_count:  # Not prune frozen Stage1.
                    continue
                if conv_count == 2:  # Stage1's 1st bn layer.
                    # Pruning 'out_planes'.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])
                elif conv_count <= 6:  # Stage1's other bn layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])

                elif conv_count == 7:  # Stage2's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])
                elif conv_count <= 11:  # Stage2's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])

                elif conv_count == 12:  # Stage3's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                elif conv_count <= 16:  # Stage3's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

                elif conv_count == 17:  # Stage4's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                elif conv_count <= 21:  # Stage4's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

            elif isinstance(module, nn.Linear) and self.args.freeze_layer != 4:
                torch_prune.ln_structured(module, name="weight", amount=pr_prob[-1], n=2, dim=1)

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        return glb_model

    def _ln_prune_r56(self, glb_model, pr_prob, remove=0):
        conv_count = 0
        down_count = 0
        for name, module in glb_model.named_modules():
            # Remove the pruning.
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in resnet.
                    conv_count += 1
                    continue

                if 'downsample' in name:
                    # The first downsample conv layer, only prune 'out_planes'.
                    if down_count == 0:
                        # Use the pruning probability in stage1(pr_prob[0]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')
                        down_count += 1
                    else:  # The other downsample conv layer.
                        # Use the pruning probability in last stage(pr_prob[down_count-1]) to prune 'in_planes'(dim=1),
                        # and the pruning probability in this stage(pr_prob[down_count]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count-1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')
                        down_count += 1
                    conv_count += 1
                    continue

                else:  # Normal conv layers in blocks.
                    if conv_count == 1:  # Stage1's 1st conv layer.
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')
                    elif conv_count <= 19:  # Stage1's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')

                    elif conv_count == 20:  # Stage2's 1st conv layer.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')
                    elif conv_count <= 38:  # Stage2's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')

                    elif conv_count == 39:  # Stage3's 1st conv layer.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')
                    elif conv_count <= 57:  # Stage3's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)
                        if remove:
                            torch_prune.remove(module, 'weight')

                    conv_count += 1
                    continue

            elif isinstance(module, nn.BatchNorm2d):
                # 'conv_count' in nn.BatchNorm2d is 1 bigger than nn.Conv2d.
                if conv_count == 1:  # The 1st bn in resnet.
                    pass
                if conv_count == 2:  # Stage1's 1st bn layer.
                    # Pruning 'out_planes'.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])
                    if remove:
                        torch_prune.remove(module, 'weight')
                elif conv_count <= 20:  # Stage1's other bn layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])
                    if remove:
                        torch_prune.remove(module, 'weight')

                elif conv_count == 21:  # Stage2's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])
                    if remove:
                        torch_prune.remove(module, 'weight')
                elif conv_count <= 39:  # Stage2's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])
                    if remove:
                        torch_prune.remove(module, 'weight')

                elif conv_count == 40:  # Stage3's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                    if remove:
                        torch_prune.remove(module, 'weight')
                elif conv_count <= 58:  # Stage3's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                    if remove:
                        torch_prune.remove(module, 'weight')

            elif isinstance(module, nn.Linear):
                torch_prune.ln_structured(module, name="weight", amount=pr_prob[-1], n=2, dim=1)
                if remove:
                    torch_prune.remove(module, 'weight')

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        return glb_model

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            if 'mask' in k:
                continue
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx, models_prune, w_global, best_acc):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        orig_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        ratios = []

        client = self.client_list[0]

        self.model_trainer.set_model_params(w_global)
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            if self.args.local_test:
                test_idx = client_idx % self.args.client_num_per_round
                self.model_trainer.set_model(models_prune[test_idx])
                params = copy.deepcopy(models_prune[test_idx].state_dict())  # params = models_prune[test_idx].state_dict()
                self.model_trainer.set_model_params(params)

            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        if test_acc > best_acc:
            best_acc = test_acc

            # save the ckpt.
            if self.args.store_ckpt:
                filename = './checkpoints/%s.pth.tar' % self.args.prefix
                model = self.model_trainer.get_model()
                model.load_state_dict(w_global)
                state = {
                    'round': round_idx + 1,
                    'best_acc': best_acc,
                    'model': model,
                }
                torch.save(state, filename)
                logging.info("Checkpoint is saved at %s" % filename)

        # pareto-ratios
        orig_acc = 0
        orig_loss = 0
        if self.args.pareto and self.args.pr_strategy == 'AD':
            orig_acc = sum(orig_metrics['num_correct']) / sum(orig_metrics['num_samples'])
            orig_loss = sum(orig_metrics['losses']) / sum(orig_metrics['num_samples'])
            logging.info("Ratio of clients: %s" % (Counter(ratios)))

        # metrics of each client.
        for i in range(len(test_metrics['num_correct'])):
            logging.info("metrics of client %s, test_acc: %s, test_loss: %s" %
                         (i, test_metrics['num_correct'][i] / test_metrics['num_samples'][i],
                          test_metrics['losses'][i] / test_metrics['num_samples'][i]))

        stats = {'training_acc': train_acc, 'training_loss': train_loss,
                 'orig_acc': orig_acc, 'orig_loss': orig_loss}
        wandb.log({"Train/Acc": train_acc, 'Orig/acc': orig_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, 'Orig/loss': orig_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss, 'best_acc': best_acc}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "Best/Acc": best_acc, "round": round_idx})
        logging.info(stats)

        return best_acc


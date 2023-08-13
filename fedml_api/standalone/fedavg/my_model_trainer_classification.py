import logging
import os

import torch
from torch import nn

from fedml_api.model.cv import SCA

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model(self):
        return self.model.cpu()

    def set_model(self, model):
        self.model = model

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        if args.dataparallel == 1:
            model = nn.DataParallel(model)
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        # model.to(device)
        model.cuda()
        model.train()

        # train and update
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().cuda()
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=False)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # x, labels = x.to(device), labels.to(device)
                x, labels = x.cuda(), labels.cuda()
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        # # Print SCA info.
        # if 'att' in args.model:
        #     scale = SCA._get_scale()
        #     print("scale len:", len(scale))
        #     # print("Sum of scale:")
        #     # print(scale)
        #     print("scale:")
        #     for s in scale:
        #         print(s.tolist()[0:10])


    def test(self, test_data, device, args):
        model = self.model

        if args.dataparallel == 1:
            model = nn.DataParallel(model)
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        # model.to(device)
        model.cuda()
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().cuda()

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                # x = x.to(device)
                # target = target.to(device)
                x = x.cuda()
                target = target.cuda()
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

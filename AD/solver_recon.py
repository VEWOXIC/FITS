import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from model.FITS import Model
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt

from thop import profile

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.95 ** ((epoch - 1) // 1))}
    # lr_adjust = {epoch: lr_ * (0.8 ** ((epoch - 1) // 1))} # for WADI only !!!!!!!!!!!!
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False,dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset=dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = Model(DotDict({'seq_len': self.win_size//self.DSR, 'enc_in': self.input_c, 'individual': self.individual,'cut_freq':self.cutfreq,'pred_len':self.win_size-self.win_size//self.DSR}))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(self.model)

        if torch.cuda.is_available():
            self.model.cuda()

    # def _get_profile(self, model):
    #     _input=torch.randn(1, self.win_size//self.DSR, self.input_c).to(self.device)
    #     macs, params = profile(model, inputs=(_input,))
    #     print('FLOPs: ', macs)
    #     print('params: ', params)
    #     return macs, params

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            # input = input_data.float().to(self.device)
            # output, series, prior, _ = self.model(input)
            
            input = input_data.float().to(self.device)[:,::self.DSR,:]
            # print(input.shape, self.win_size//4)

            output, _ = self.model(input)

            ###########################

            rec_loss = self.criterion(output, input_data.float().to(self.device))

            loss_1.append((rec_loss).item())
            
            
            # loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # self._get_profile(self.model)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)[:,::self.DSR,:] # downsample
                # print(input.shape, self.win_size//4)

                output, _ = self.model(input)

                ###########################

                rec_loss = self.criterion(output, input_data.float().to(self.device))

                loss1_list.append((rec_loss).item())
                loss1 = rec_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s, loss: {:.4f}'.format(speed, left_time,rec_loss.item()))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                # loss2.backward()
                self.optimizer.step()
        
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            # vali_loss1 = self.vali(self.test_loader)
            vali_loss1 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1,  self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if epoch <=25:
                adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        # torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            # input = input_data.float().to(self.device)
            # output, series, prior, _ = self.model(input)
            
            input = input_data.float().to(self.device)[:,::self.DSR,:]
            # print(input.shape, self.win_size//4)

            output, _ = self.model(input)

            ###########################
            for u in range(output.shape[0]):
                rec_loss = self.criterion(output[u], input_data[u].float().to(self.device))


            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = rec_loss.unsqueeze(0)
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
        # print(attens_energy)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.vali_loader):
            input = input_data.float().to(self.device)[:,::self.DSR,:]
            # print(input.shape, self.win_size//4)

            output, _ = self.model(input)

            ###########################

            for u in range(output.shape[0]):
                rec_loss = self.criterion(output[u], input_data[u].float().to(self.device))


            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = rec_loss.unsqueeze(0)
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)[:,::self.DSR,:]
            # print(input.shape, self.win_size//4)

            output, _ = self.model(input)

            ###########################

            for u in range(output.shape[0]):
                rec_loss = self.criterion(output[u], input_data[u].float().to(self.device))
                if self.plot:
                    plt.plot(input_data[u,:,11].detach().cpu().numpy())
                    plt.plot(output[u,:,11].detach().cpu().numpy())
                    plt.title(rec_loss.item())
                    plt.savefig('pics/test{}.png'.format(i*self.batch_size+u))
                    plt.clf()


            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = rec_loss.unsqueeze(0)
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append((torch.sum(labels)>0).unsqueeze(0))
            # print(cri,labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        print(test_energy.shape, test_labels.shape)
        plt.plot(test_energy)
        plt.plot(test_labels.astype(int)*100)
        plt.savefig('test_energy.png')

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score

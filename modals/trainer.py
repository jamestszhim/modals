import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.blstm import BiLSTM
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from utility import mixup_criterion, mixup_data

from modals.data_util import get_text_dataloaders
from modals.policy import PolicyManager, RawPolicy

if torch.cuda.is_available():
    import modals.augmentation_transforms as aug_trans
else:
    import modals.augmentation_transforms_cpu as aug_trans

from modals.custom_ops import (HardestNegativeTripletSelector,
                               RandomNegativeTripletSelector,
                               SemihardNegativeTripletSelector)
from modals.losses import (OnlineTripletLoss, adverserial_loss,
                           discriminator_loss)


def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f' |Trainable parameters: {temp}')


def build_model(model_name, vocab, n_class, z_size=2):
    net = None
    if model_name == 'blstm':
        config = {'n_vocab': len(vocab),
                  'n_embed': 300,
                  'emb': vocab.vectors,
                  'n_hidden': 256,
                  'n_output': n_class,
                  'n_layers': 2,
                  'pad_idx': vocab.stoi['<pad>'],
                  'b_dir': True,
                  'rnn_drop': 0.2,
                  'fc_drop': 0.5}
        net = BiLSTM(config)
        z_size = 256
    else:
        ValueError(f'Invalid model name={model_name}')

    print('\n### Model ###')
    print(f'=> {model_name}')
    count_parameters(net)

    return net, z_size, model_name


class Discriminator(nn.Module):
    def __init__(self, z_size):
        super(Discriminator, self).__init__()
        self.z_size = z_size
        self.fc1 = nn.Linear(z_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class TextModelTrainer(object):

    def __init__(self, hparams, name=''):
        self.hparams = hparams
        print(hparams)

        self.name = name

        random.seed(0)
        self.train_loader, self.valid_loader, self.test_loader, self.classes, self.vocab = get_text_dataloaders(
            hparams['dataset_name'], valid_size=hparams['valid_size'], batch_size=hparams['batch_size'],
            subtrain_ratio=hparams['subtrain_ratio'], dataroot=hparams['dataset_dir'])
        random.seed()

        self.device = torch.device(
            hparams['gpu_device'] if torch.cuda.is_available() else 'cpu')
        print()
        print('### Device ###')
        print(self.device)
        self.net, self.z_size, self.file_name = build_model(
            hparams['model_name'], self.vocab, len(self.classes))
        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        if hparams['mode'] in ['train', 'search']:
            self.optimizer = optim.Adam(self.net.parameters(), 0.001)
            self.loss_dict = {'train': [], 'valid': []}

            if hparams['use_modals']:
                print("\n=> ### Policy ###")
                # print(f'  |hp_policy: {hparams['hp_policy']}')
                # print(f'  |policy_path: {hparams['policy_path']}')
                raw_policy = RawPolicy(mode=hparams['mode'], num_epochs=hparams['num_epochs'],
                                       hp_policy=hparams['hp_policy'], policy_path=hparams['policy_path'])
                transformations = aug_trans
                self.pm = PolicyManager(
                    transformations, raw_policy, len(self.classes), self.device)

            print("\n### Loss ###")
            print('Classification Loss')

            if hparams['mixup']:
                print('Mixup')

            if hparams['enforce_prior']:
                print('Adversarial Loss')
                self.EPS = 1e-15
                self.D = Discriminator(self.z_size)
                self.D = self.D.to(self.device)
                self.D_optimizer = optim.SGD(self.D.parameters(), lr=0.01,
                                             momentum=hparams['momentum'], weight_decay=hparams['wd'])
                # self.G_optimizer = optim.Adam(self.net.parameters(), lr=0.001)

            if hparams['metric_learning']:
                margin = hparams['metric_margin']
                metric_loss = hparams["metric_loss"]
                metric_weight = hparams["metric_weight"]
                print(
                    f"Metric Loss (margin: {margin} loss: {metric_loss} weight: {metric_weight})")

                self.M_optimizer = optim.SGD(
                    self.net.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-8)
                self.metric_weight = hparams['metric_weight']

                if metric_loss == 'random':
                    self.metric_loss = OnlineTripletLoss(
                        margin, RandomNegativeTripletSelector(margin))
                elif metric_loss == 'hardest':
                    self.metric_loss = OnlineTripletLoss(
                        margin, HardestNegativeTripletSelector(margin))
                elif metric_loss == 'semihard':
                    self.metric_loss = OnlineTripletLoss(
                        margin, SemihardNegativeTripletSelector(margin))

    def reset_model(self, z_size=256):
        # tunable z_size only use for visualization
        # if blstm is used, it is automatically 256
        self.net, self.z_size, self.file_name = build_model(
            self.hparams['model_name'], self.vocab, len(self.classes), z_size)
        self.net = self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), 0.001)
        self.loss_dict = {'train': [], 'valid': []}

    def reset_discriminator(self, z_size=256):
        self.D = Discriminator(z_size)
        self.D = self.D.to(self.device)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=0.01,
                                     momentum=self.hparams['momentum'], weight_decay=self.hparams['wd'])

    def update_policy(self, policy):
        raw_policy = RawPolicy(mode='train', num_epochs=1,
                               hp_policy=policy, policy_path=None)
        self.pm.update_policy(raw_policy)

    def _train(self, cur_epoch):
        self.net.train()
        self.net.training = True
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(self.train_loader))  # cosine learning rate
        train_losses = 0.0
        clf_losses = 0.0
        metric_losses = 0.0
        d_losses = 0.0
        g_losses = 0.0
        correct = 0
        total = 0
        n_batch = len(self.train_loader)

        print(f'\n=> Training Epoch #{cur_epoch}')
        for batch_idx, batch in enumerate(self.train_loader):

            inputs, seq_lens, labels = batch.text[0].to(
                self.device), batch.text[1].to(self.device), batch.label.to(self.device)

            # if self.hparams['dataset_name'] == 'sst2':
            labels -= 1  # because I binarized the data

            seed_features = self.net.extract_features(inputs, seq_lens)
            features = seed_features

            if self.hparams['manifold_mixup']:
                features, targets_a, targets_b, lam = mixup_data(features, labels,
                                                                 0.2, use_cuda=True)
                features, targets_a, targets_b = map(Variable, (features,
                                                                targets_a, targets_b))
            # apply pba transformation
            if self.hparams['use_modals']:
                features = self.pm.apply_policy(
                    features, labels, cur_epoch, batch_idx, verbose=1).to(self.device)

            outputs = self.net.classify(features)  # Forward Propagation

            if self.hparams['mixup']:
                inputs, targets_a, targets_b, lam = mixup_data(outputs, labels,
                                                               self.hparams['alpha'], use_cuda=True)
                inputs, targets_a, targets_b = map(Variable, (outputs,
                                                              targets_a, targets_b))
            # freeze D
            if self.hparams['enforce_prior']:
                for p in self.D.parameters():
                    p.requires_grad = False

            # classification loss
            if self.hparams['mixup'] or self.hparams['manifold_mixup']:
                c_loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam)
            else:
                c_loss = self.criterion(outputs, labels)  # Loss
            clf_losses += c_loss.item()

            # total loss
            loss = c_loss
            if self.hparams['metric_learning']:
                m_loss = self.metric_loss(seed_features, labels)[0]
                metric_losses += m_loss.item()
                loss = self.metric_weight * m_loss + \
                    (1-self.metric_weight) * c_loss

            train_losses += loss.item()

            if self.hparams['enforce_prior']:
                # Regularizer update
                # freeze D
                for p in self.D.parameters():
                    p.requires_grad = False
                self.net.train()
                d_fake = self.D(features)
                g_loss = self.hparams['prior_weight'] * \
                    adverserial_loss(d_fake, self.EPS)
                g_losses += g_loss.item()
                loss += g_loss

            self.optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimizer.step()  # Optimizer update

            if self.hparams['enforce_prior']:
                # Discriminator update
                for p in self.D.parameters():
                    p.requires_grad = True

                features = self.net.extract_features(inputs, seq_lens)
                d_real = self.D(torch.randn(features.size()).to(self.device))
                d_fake = self.D(F.softmax(features, dim=0))
                d_loss = discriminator_loss(d_real, d_fake, self.EPS)
                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()
                d_losses += d_loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if self.hparams['mixup']:
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            else:
                correct += (predicted == labels).sum().item()

        # step
        step = (cur_epoch-1)*(len(self.train_loader)) + batch_idx
        total_steps = self.hparams['num_epochs']*len(self.train_loader)

        # logs
        display = f'| Epoch [{cur_epoch}/{self.hparams["num_epochs"]}]\tIter[{step}/{total_steps}]\tLoss: {train_losses/n_batch:.4f}\tAcc@1: {correct/total:.4f}\tclf_loss: {clf_losses/n_batch:.4f}'
        if self.hparams['enforce_prior']:
            display += f'\td_loss: {d_losses/n_batch:.4f}\tg_loss: {g_losses/n_batch:.4f}'
        if self.hparams['metric_learning']:
            display += f'\tmetric_loss: {metric_losses/n_batch:.4f}'
        print(display)

        return correct/total, train_losses/total

    def _test(self, cur_epoch, mode):
        self.net.eval()
        self.net.training = False
        correct = 0
        total = 0
        test_loss = 0.0
        data_loader = self.valid_loader if mode == 'valid' else self.test_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, seq_lens, labels = batch.text[0].to(
                    self.device), batch.text[1].to(self.device), batch.label.to(self.device)

                # if self.hparams['dataset_name'] == 'sst2':
                labels -= 1  # because I binarized the data

                outputs = self.net(inputs, seq_lens)
                loss = self.criterion(outputs, labels)  # Loss
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                torch.cuda.empty_cache()

            print(
                f'| ({mode}) Epoch #{cur_epoch}\t Loss: {test_loss/total:.4f}\t Acc@1: {correct/total:.4f}')

        return correct/total, test_loss/total

    def run_model(self, epoch):
        if self.hparams['use_modals']:
            self.pm.reset_text_data_pool(
                self.net, self.train_loader, self.hparams['temperature'], self.hparams['distance_metric'], self.hparams['dataset_name'])

        train_acc, tl = self._train(epoch)
        self.loss_dict['train'].append(tl)

        if self.hparams['valid_size'] > 0:
            val_acc, vl = self._test(epoch, mode='valid')
            self.loss_dict['valid'].append(vl)
        else:
            val_acc = 0.0

        return train_acc, val_acc

    # for benchmark
    def save_checkpoint(self, ckpt_dir, epoch):
        path = os.path.join(
            ckpt_dir, self.hparams['dataset_name'], f'{self.name}_{self.file_name}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path

    # for ray
    def save_model(self, ckpt_dir, epoch):
        # save the checkpoint.
        print(self.file_name)
        print(ckpt_dir)
        path = os.path.join(ckpt_dir, self.file_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'state': self.net.state_dict(),
                    'epoch': epoch,
                    'loss': self.loss_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}, path)

        print(f'=> saved the model {self.file_name} to {path}')
        return path

    def load_model(self, ckpt):
        # load the checkpoint.
        # path = os.path.join(ckpt_dir, self.model_name)
        # map_location='cuda:0')
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['state'])
        self.loss_dict = checkpoint['loss']
        if self.hparams['mode'] != 'test':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'=> loaded checkpoint of {self.file_name} from {ckpt}')
        return checkpoint['epoch'], checkpoint['loss']

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        new_policy = RawPolicy(mode=self.hparams['mode'], num_epochs=self.hparams['num_epochs'],
                               hp_policy=self.hparams['hp_policy'])
        self.pm.update_policy(new_policy)
        return


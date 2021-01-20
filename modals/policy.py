import json
import math

import numpy as np
import torch
import torch.nn as nn
from modals.custom_ops import cosine
from sklearn.metrics.pairwise import euclidean_distances


def parse_policy_log(policy_file):
    ### When Magic happends ###
    pbt_policy_file = open(policy_file, 'r').readlines()

    perturb_events = []
    for perturb_event in pbt_policy_file:
        event = json.loads(perturb_event)
        perturb_events.append(event)

    initial_policy = perturb_events[0][4]['hp_policy']
    policy_num_epochs = perturb_events[0][4]['num_epochs']

    '''
    epoch    0  15  20  35  100
    policy  p0  p1  p2  p3
    =>
    epoch   15-0  20-15  35-20  100-35
    policy  p0    p1     p2     p3
    '''

    # the epoch policy changed
    perturb_epoch = [0]

    # the policy changed
    perturb_policy = [initial_policy]  # initial policy i.e. [0] policy

    for event in perturb_events:
        perturb_epoch.append(event[3])
        perturb_policy.append(event[5]['hp_policy'])

    perturb_epoch.append(policy_num_epochs)

    # how many times running_policy[i] is ran
    n_repeats = [0] * (len(perturb_epoch)-1)

    for i in range(len(n_repeats)):
        n_repeats[i] = perturb_epoch[i+1] - perturb_epoch[i]

    assert len(perturb_policy) == len(n_repeats)
    assert sum(n_repeats) == policy_num_epochs

    return (n_repeats, perturb_policy)


class RawPolicy(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, mode, num_epochs, hp_policy=None, policy_path=None):
        """
        search: pba, must be single
        train/test: using a pba schedule to train/test a child model, 
        can be a schedule or a single. However, the piority is given
        to policy_path.
        """
        assert mode in ['search', 'train', 'visualize']
        if mode == 'search' or mode == 'visualize':
            assert hp_policy is not None
            self.type = 'single'
            self.emb = hp_policy
        else:
            if policy_path is not None:
                # Parse policy form pbt_policy_{i}.txt
                n_repeats, raw_policies = parse_policy_log(policy_path)
                if num_epochs != sum(n_repeats):
                    print('Interpolating policy')
                    ratio = num_epochs / sum(n_repeats)
                    n_repeats = [math.floor(n * ratio) for n in n_repeats]
                    n_pad = num_epochs - sum(n_repeats)
                    n_repeats[-1] += n_pad
                    assert num_epochs == sum(n_repeats)

                # Unroll a policy
                self.schedule = np.repeat(
                    raw_policies, n_repeats, axis=0)
                self.type = 'schedule'

            elif hp_policy is not None:
                if isinstance(hp_policy[0], list):
                    # provided schdule must match epochs
                    assert len(hp_policy) == num_epochs
                    self.type = 'schedule'
                    self.schedule = hp_policy
                else:
                    self.type = 'single'
                    self.emb = hp_policy
            else:
                raise ValueError('You must provide hp_policy or policy path')


class PolicyManager(object):
    """Manage policy."""

    def __init__(self, aug_trans, raw_policy, num_classes, device):
        self.num_xform = aug_trans.NUM_HP_TRANSFORM
        self.xform_names = aug_trans.HP_TRANSFORM_NAMES
        self.apply_policy_from_pool = aug_trans.apply_policy_from_pool
        self.policy = None
        self.update_policy(raw_policy)
        self.num_classes = num_classes
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def apply_pba(self, images):
        return

    def update_policy(self, raw_policy):
        self.raw_policy = raw_policy
        self.policy = self.parse_policy(raw_policy)
        if raw_policy.type == 'single':
            print(f'Updated hp policy: {self.policy}')

    def parse_policy(self, raw_policy):
        if raw_policy.type == 'single':
            return self._parse_one_policy(raw_policy.emb)
        elif raw_policy.type == 'schedule':
            policy = []
            for one_emb in raw_policy.schedule:
                policy.append(self._parse_one_policy(one_emb))
            return policy

    def _parse_one_policy(self, emb):
        assert len(
            emb) == 2*2*self.num_xform, f'raw policy was: {len(emb)}, supposed to be: {2*2*self.num_xform}'
        one_policy = []
        for i, xform in enumerate(list(self.xform_names)*2):
            one_policy.append((xform, emb[2 * i] / 10., emb[2 * i + 1]))
        return one_policy

    def apply_policy(self, imgs, labels, epoch, batch_idx, verbose=1):
        inner_verbose = True if verbose > 1 else False
        cur_epoch = max(epoch-1, 0)
        running_policy = self.policy if self.raw_policy.type == 'single' else self.policy[
            cur_epoch]
        x_imgs = imgs.new_empty(imgs.shape)

        for i, label in enumerate(labels):
            class_info = {'pool': self.feat_pool[self.idx_by_class[label]],
                          'weights': self.img_weights_by_class[label],
                          'mean': self.mean_by_class[label],
                          'sd': self.sd_by_class[label]}
            x_img, _ = self.apply_policy_from_pool(running_policy, imgs[i],
                                                   class_info, inner_verbose)
            x_imgs[i] = x_img

        if batch_idx == 0 and verbose > 0:
            print(f'Applying policy {running_policy}')

        return x_imgs

    def print_data_pool_report(self, cc, cmu):
        print("### Data Pool Report ###")
        print(f'Cluster Closeness: {np.around(cc, 4)}')
        print("Cluster mean distance:")
        print(euclidean_distances(cmu))
        print()

    def reset_text_data_pool(self, encoder, dataloader, temperature, weight_metric, dataset, verbose=False, return_report=False):
        # compute the sd and mean of each lass
        # save the features into a pool
        feat_pool = []
        label_pool = []
        loss_pool = []
        feat_dim = 0
        encoder.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs, seq_lens, labels = batch.text[0].to(
                    self.device), batch.text[1].to(self.device), batch.label.to(self.device)

                if dataset == 'sst2' or dataset == 'trec':
                    labels -= 1  # because I binarized the data

                features = encoder.extract_features(inputs, seq_lens)
                outputs = encoder.classify(features)
                loss = self.criterion(outputs, labels.to(self.device))
                for i in range(len(labels)):
                    feat_pool.append(features[i].cpu())
                    label_pool.append(labels[i].cpu())
                    loss_pool.append(loss[i].cpu())

        feat_dim = feat_pool[0].shape[0]
        self.feat_pool = torch.stack(
            feat_pool).reshape(-1, feat_dim).double()  # list of all images
        label_pool = torch.stack(label_pool).reshape(-1)
        loss_pool = torch.stack(loss_pool).reshape(-1)

        self.idx_by_class = []  # [0]: [1,4,2,...] <-index that belongs to class 0
        self.mean_by_class = []  # [0]: mean of class 0
        self.sd_by_class = []  # [0]: sd of class 0
        self.img_weights_by_class = []  # [0]: weights of the images in class 0

        cluster_closeness = []
        class_means = []

        for i in range(self.num_classes):
            img_idxs = torch.where(label_pool == i)[0]
            self.idx_by_class.append(img_idxs)
            class_imgs = self.feat_pool[img_idxs]
            class_loss = loss_pool[img_idxs]

            class_mean = class_imgs.mean(0)
            self.mean_by_class.append(class_mean)
            img_distances = np.linalg.norm(
                class_imgs-class_mean, ord=2, axis=1)  # distance from mean

            class_means.append(class_mean.numpy())
            # in-class pari-wise distance
            icpd = euclidean_distances(class_imgs)
            aicpd = np.sum(icpd)/(len(icpd)*(len(icpd)-1))
            cluster_closeness.append(aicpd)

            if weight_metric == 'l2':
                # weight by l2 distance
                img_weights = img_distances
                img_weights -= img_weights.max(0)  # numerical stability
            elif weight_metric == 'cosine':
                # weight by consine distance
                img_weights = cosine(class_imgs, class_mean)
            elif weight_metric == 'loss':
                # weight by loss
                img_weights = class_loss
                img_weights -= torch.max(img_weights)
            elif weight_metric == 'same':
                # uniform
                img_weights = np.ones(len(class_imgs))/len(class_imgs)

            img_weights = np.exp(img_weights/temperature)
            img_weights = img_weights/img_weights.sum()  # normalize

            self.img_weights_by_class.append(np.nan_to_num(img_weights))
            self.sd_by_class.append(class_imgs.std(0))

        if verbose:
            self.print_data_pool_report(cluster_closeness, class_means)

        if return_report:
            cpd = euclidean_distances(class_means)
            acpd = np.sum(cpd)/(len(cpd)*(len(cpd)-1))
            return np.mean(cluster_closeness), acpd

    def print_policy(self):
        print(self.policy)

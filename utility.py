import os
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb+') as fo:
        dict = pickle.load(fo, encoding='utf-8')
    return dict

def get_image_stats(dataset):

    MEANS = {
        'cifar10': (0.49139968, 0.48215841, 0.44653091),
        'reduced_cifar10': (0.49056774, 0.48116026, 0.44726052),
        'cifar100': (0.50707516, 0.48654887, 0.44091784),
        'reduced_svhn': (0.45163885, 0.4557915, 0.48093327),
        'ablation_svhn': (0.20385217, 0.20957996, 0.20804394),
        'svhn': (0.43090966, 0.4302428, 0.44634357)
    }
    STDS = {
        'cifar10': (0.24703223, 0.24348513, 0.26158784),
        'reduced_cifar10': (0.24710728, 0.24451308, 0.26235099),
        'cifar100': (0.26733429, 0.25643846, 0.27615047),
        'reduced_svhn': (0.20385217, 0.20957996, 0.20804394),
        'ablation_svhn': (0.20385217, 0.20957996, 0.20804394),
        'svhn': (0.19652855, 0.19832038, 0.19942076)
    }

    return MEANS[dataset], STDS[dataset]


def imshow(img, dataset, normalize=False):
    img = img.clone()
    if normalize:
        m, s = get_image_stats(dataset)
        for t, m, s in zip(img, m, s):
            t.mul_(s).add_(m)

    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def cosine_lr(learning_rate, cur_step, total_step):
    """Cosine Learning rate.

    Args:
      learning_rate: Initial learning rate.
      epoch: Current epoch we are one. This is one based.
      iteration: Current batch in this epoch.
      batches_per_epoch: Batches per epoch.
      total_epochs: Total epochs you are training for.

    Returns:
      The learning rate to be used for this current batch.
    """
    # t_total = total_epochs * batches_per_epoch
    # t_cur = float(epoch * batches_per_epoch + iteration)
    return 0.5 * learning_rate * (1 + np.cos(np.pi * cur_step / total_step))


def get_lr(learning_rate, iteration=None, total_iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None
    lr = cosine_lr(learning_rate, iteration, total_iteration)
    return lr


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def save_checkpoint(model, name, model_dir, epoch, loss_dict):
    path = os.path.join(model_dir, name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(),
                'epoch': epoch, 'loss': loss_dict}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=name, path=path
    ))


def load_checkpoint(model, name, model_dir):
    path = os.path.join(model_dir, name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def export_feature(dataloader, net, save_path, device):

    img_features = []
    img_labels = []

    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        x = net.extract_features(images)
        features = x.view(x.size(0), -1)
        img_features.extend(features.detach().cpu().numpy())
        img_labels.extend(labels.cpu().numpy())

    img_features = np.array(img_features)
    img_labels = np.array(img_labels)
    print(img_features.shape)
    print(img_labels.shape)
    np.save(f'feat_{save_path}', img_features)
    np.save(f'label_{save_path}', img_features)

def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))

    return _to_tensor

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

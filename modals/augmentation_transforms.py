import collections
import copy
import random

import numpy as np
import torch

from modals.custom_ops import cosine

PARAMETER_MAX = 10


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    x = float(level) * maxval / PARAMETER_MAX

    return torch.as_tensor(x)


def apply_policy_from_pool(policy, img, img_pool, verbose=0):
    """Apply the `policy` to the sentence.

    Args:
    policy: A list of tuples with the form (name, probability, magnitude) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `magnitude` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
    verbose: 0: no log
             1: text log
             2: visualization log

    Returns:
    The result of applying `policy` to `sentence`.
    """
    label_img_pool = img_pool
    display = '=> '
    count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.7, 0.1, 0.0])
    support_idxs = []
    ximg = img
    if count != 0:
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, magnitude = xform
            assert 0. <= probability <= 1.
            assert 0 <= magnitude <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].transformer(
                probability, magnitude)
            (ximg, support_idx), res = xform_fn(ximg, img_pool) # 1st: (img, support)
            if verbose > 0 and res:
                display += f"Op: {name}, Magnitude: {magnitude}, Prob: {probability} "
                if verbose > 1:
                    support_idxs.append(support_idx)
            count -= res
            assert count >= 0
            if count == 0:
                break
        if verbose:
            print(display)
        return ximg, support_idxs
    else:
        return img, []


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, img, label_img_pool):
        return self.f(img, label_img_pool)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def transformer(self, probability, magnitude):

        def return_function(img, label_img_pool):
            res = False
            s = []
            if random.random() < probability:
                img, s = self.xform(img, label_img_pool, magnitude)
                res = True
            return (img, s), res

        name = self.name + '({:.1f},{})'.format(probability, magnitude)
        return TransformFunction(return_function, name)

    def do_transform(self, img, label_img_pool, magnitude):
        f = self.transformer(PARAMETER_MAX, magnitude)
        return f(img, label_img_pool)


def _interpolate(img, class_info, magnitude):
    ''' this function interpolates target imgage with a pool of other images
    using a magnitude
    img: a 1D numpy arrays
    img_pool: a 2D numpy array'''
    m = float_parameter(magnitude, 1)
    x = img
    p = class_info['weights']
    if len(p)<1:
        return img, []
    k = max(1, int(len(class_info['pool']) * 0.05))
    idxs = np.random.choice(len(class_info['pool']), k, p=p) #choose points near to the boundary
    distances = cosine(class_info['pool'][idxs]-class_info['mean'], x.detach().cpu().view(-1)-class_info['mean']) #but not too far from the seed
    idx = idxs[np.argmax(distances)]
    y = class_info['pool'][idx]
    x_hat = (y.cuda()-x)*m + x
    return x_hat, [idx]


interpolate = TransformT('Interpolate', _interpolate)


def _extrapolate(img, class_info, magnitude):
    ''' this function extrapolate target imgage with a pool of other images
    using a magnitude
    img: a 1D numpy arrays
    img_pool: a 2D numpy array'''

    m = float_parameter(magnitude, 1)
    x = img
    mu = class_info['mean']
    x_hat = (x-mu.cuda())*m + x
    return x_hat, []


extrapolate = TransformT('Extrapolate', _extrapolate)


def _linearpolate(img, class_info, magnitude):
    ''' this function linear move target imgage with a pool of other images
    using a magnitude
    img: a 1D numpy arrays
    img_pool: a 2D numpy array'''

    m = float_parameter(magnitude, 1)
    x = img
    if len(class_info['pool']) < 2:
        return x, [0,0]
    idx1, idx2 = random.sample(range(len(class_info['pool'])), 2)
    y1, y2 = class_info['pool'][idx1], class_info['pool'][idx2]
    x_hat = (y1.cuda()-y2.cuda())*m + x
    return x_hat, [idx1, idx2]


linear_polate = TransformT('LinearPolate', _linearpolate)


def _resample(img, class_info, magnitude):
    x = img
    m = float_parameter(magnitude, 1)
    noise = torch.randn(img.size()).cuda()
    x_hat = x+noise*class_info['sd'].cuda()*m
    return x_hat, []


resample = TransformT('Resample', _resample)

HP_TRANSFORMS = [
    interpolate,
    extrapolate,
    linear_polate,
    resample
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)

# !/usr/bin/env python3
"""
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""
import numpy as np


def remove_row_mean(in_content):
    """
    Remove the mean along the inline (x) direction
    """
    if in_content.dtype == np.uint8:
        in_content = in_content / 255.

    return in_content - np.mean(in_content, 2)[:,:,None]



def rectify_column(in_content, margin=20):
    """
    rectify the ground interface
    """
    temp = in_content

    for bsc_idx, bscan in enumerate(in_content):
        temp[bsc_idx] = np.roll(bscan, -np.argmax(bscan, axis=0) + margin, axis=1)

    return temp

def cubic_dynamics(in_content, th=0.05):
    """
    Use a cubic power for expanding the dynamics of the data
    :param in_content: input volume
    :param th: value of the input signal to be mapped to 1 (i.e. threshold of the cubic curve)
    :return: processed volume
    """
    temp = in_content - np.mean(in_content, (1,2))[:,None,None]
    temp /= th
    temp = temp**3

    return temp

def normalize_silvia(in_content, sigmaS=14.5, meanS=121.8, alpha=13):
    """
    normalize wrt the synthetic data
    :param in_content: input volume
    :param sigmaS: standard deviation of the synthetic data
    :param meanS: mean of the synthetic data
    :param alpha: it's magic
    :return: processed volume
    """
    sigmaIn = np.std(in_content, (1,2))
    return meanS + sigmaS / sigmaIn[:,None,None] / alpha * (in_content - np.mean(in_content, (1,2))[:,None,None])

def normalize_silvia_hard(in_content, sigmaS=14.5, meanS=121.8, alpha=13):
    return (in_content - 0.3267) / 0.0084 * sigmaS / alpha + meanS

def normalize(in_content, in_min=None, in_max=None):
    if in_min is None and in_max is None:
        in_min = np.min(in_content)
        in_max = np.max(in_content)
    in_content = (in_content - in_min) / (in_max - in_min)
    in_content = in_content*2 - 1
    return in_content, in_min, in_max

def apply_transform(in_content, transform:callable):
    return transform(in_content)

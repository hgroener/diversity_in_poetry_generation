from datasets import load_from_disk
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import wasserstein_distance
import numpy as np
import json
import argparse
import os


# chi square test
def get_contigents_len(ds1, ds2):
    """
    Generates a contingency table for two discrete length samples
    
    Parameters:
    ----------
    ds1 : First dataset
    ds2 : Second dataset

    Returns:
    -------
    Array representing a contingency table
    """
    len1 = ds1['length']
    len2 = ds2['length']
    minimum = np.min(len1 + len2)
    maximum = np.max(len1 + len2)
    cont1 = []
    cont2 = []
    for i in range(minimum, maximum+1):
        if i in len1 or i in len2:
            cont1.append(len1.count(i))
            cont2.append(len2.count(i))
    return np.array([cont1, cont2])


# returns p-value of the chi square test
def chi_square_len(path1, path2):
    """
    Performs a chi square test contingency test
    
    Parameters:
    ----------
    path1 : Path to first dataset
    path2 : Path to second dataset

    Returns:
    -------
    p-value of the chi square test
    """
    ds1 = load_from_disk(path1)
    ds2 = load_from_disk(path2)
    c = get_contigents_len(ds1, ds2)
    return chi2_contingency(c)[1]


def histogram_intersection(path1, path2):
    """
    Calculates the shared area of two normalized histograms
    
    Parameters:
    ----------
    path1 : Path to first dataset
    path2 : Path to second dataset

    Returns:
    -------
    intersection : Portion of the shared are (a value between 0 and 1)
    """
    ds1 = load_from_disk(path1)
    ds2 = load_from_disk(path2)
    len1 = ds1['length']
    len2 = ds2['length']
    minimum = np.min([np.min(len1), np.min(len2)])
    maximum = np.max([np.max(len1), np.max(len2)])
    hist_1, _ = np.histogram(len1, bins=30, range=[minimum, maximum], density=True)
    hist_2, _ = np.histogram(len2, bins=30, range=[minimum, maximum], density=True)
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


def mean_sd(path):
    """
    Calculates the mean and the standard deviation for a discrete length dataset
    
    Parameters:
    ----------
    path  : Path to dataset

    Returns:
    -------
    mean : Mean length
    std : Standard deviation of length
    """
    ds = load_from_disk(path)
    length = ds['length']
    mean = np.mean(length)
    std = np.std(length)
    return mean, std


def min_max(path):
    """
    Calculates the minimum and the maximum in a discrete length dataset
    
    Parameters:
    ----------
    path  : Path to dataset

    Returns:
    -------
    min : Minimum
    max: Maximum
    """
    ds = load_from_disk(path)
    length = ds['length']
    return np.min(length), np.max(length)


def wasserstein(path1, path2):
    """
    Calculates the Wasserstain distance (L1 norm) between two discrete samples
    
    Parameters:
    ----------
    path1 : Path to first dataset
    path2 : Path to second dataset

    Returns:
    -------
    wasserstein_distance : Calculated norm
    """
    ds1 = load_from_disk(path1)
    ds2 = load_from_disk(path2)
    length1 = ds1['length']
    length2 = ds2['length']
    return wasserstein_distance(length1, length2)


def plot_len_figures(sample_path):
    """
    Generates a length histogram for a given sample
    
    Parameters:
    ----------
    sample_path : Path to dataset

    Returns:
    -------
    fig : Figure object representing the histogram of length values
    """
    
    ds = load_from_disk(sample_path)
    length = ds['length']
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.style.use('seaborn-whitegrid')
    plt.hist(length, bins='auto', facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, density=True)
    plt.xlabel("Quatrain Length", fontsize=17)
    plt.ylabel("Densitiy", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return fig


def process_training_data(quatrain_path):
    """
    Calculates different metrics and plots a histogram for the length values of a dateset
    
    Parameters:
    ----------
    quatrain_path : Path to dataset

    Returns:
    -------
    res : Dictionary containing metric values
    fig : Figure object representing the histogram of length values
    """
    quatrain = load_from_disk(quatrain_path)
    length = quatrain['length']
    res = {}
    
    res['mean'] = float(np.mean(length))
    res['standard_deviation'] = float(np.std(length))
    res['min'] = int(np.min(length))
    res['max'] = int(np.max(length))

    fig = plt.figure(figsize=(5,4))
    plt.style.use('seaborn-whitegrid')
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.hist(length, bins=30, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, density=True)
    plt.xlabel("Quatrain Length", fontsize=17)
    plt.ylabel("Densitiy", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return res, fig


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_only', action="store_true")
    parser.add_argument('--sample_path', type=str)
    parser.add_argument('--quatrain_path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--out_name', type=str)

    args = parser.parse_args()

    if args.out_path:
        # create save path if non existent
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)

    if args.training_data_only == True:
        
        res, fig = process_training_data(args.quatrain_path)
                
        fig.savefig(args.out_path + '/' + args.lang + '-Quatrain.png', dpi=100)
        with open(args.out_path + '/' + args.lang + '-QuaTrain.json', 'w') as f:
            json.dump(res, f)

    else:
        value_dict = {}

        m, sd = mean_sd(args.sample_path)
        minimum, maximum = min_max(args.sample_path)

        hist_intersection = histogram_intersection(args.quatrain_path, args.sample_path)
        wasserstein_distance = wasserstein(args.quatrain_path, args.sample_path)

        value_dict['mean'] = float(m)
        value_dict['standard_deviation'] = float(sd)
        value_dict['min'] = int(minimum)
        value_dict['max'] = int(maximum)
        value_dict['histogram_intersection'] = float(hist_intersection)
        value_dict['wasserstein_distance'] = float(wasserstein_distance)
        
        fig = plot_len_figures(args.sample_path)

        fig.savefig(args.out_path + '/' + args.lang + '-hist-' + args.out_name + '.png' , dpi=100)
        with open(args.out_path + '/' + args.lang + '-metrics-' + args.out_name + '.json', 'w') as f:
            json.dump(value_dict, f)
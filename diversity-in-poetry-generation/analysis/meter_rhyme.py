from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
import os
import pickle
import random
import sys
import argparse
import pprint

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


def chi2_test(path1, path2, attribute):
    
    # load stats files
    with open(path1, 'rb') as f:
        ds1 = pickle.load(f)
    with open(path2, 'rb') as d:
        ds2 = pickle.load(d)
    c1 = list(ds1[attribute].values())
    #c1 = random.sample(d1, 1500)
    c2 = list(ds2[attribute].values())
    c1 = random.sample(c1, len(c2))
    p = chi2_contingency([c1,c2])[1]
    return p


#only for rhyme
def plot_stacked(rhyme, reps, length):
    rhyme = {k: v / length for k, v in rhyme.items()}
    reps = {k: v / length for k, v in reps.items()}
    reps['ABCD'] = 0
    diff = {key: rhyme[key] - reps.get(key, 0) for key in rhyme}
    fig = plt.figure(figsize=(5, 4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.2)
    plt.style.use('seaborn-whitegrid')
    plt.xticks(rotation=90)
    plt.bar(diff.keys(), list(diff.values()), color='#2ab0ff', label='Rhymes')
    plt.bar(reps.keys(), list(reps.values()), bottom=list(diff.values()), color='r', label='Repetitions')
    #plt.xlabel("Rhyme")
    plt.ylabel("Density", fontsize=17)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return diff, reps, fig


def meter_rhyme_metrics(sample_path, training_data_path):

    res = {}

    # load sample stats
    sample_path = sample_path + '/stats.pkl'

    with open(sample_path, 'rb') as f:
        stats = pickle.load(f)

    rhyme = stats['rhyme']
    meter = stats['meter']
    length = stats['length']
    reps = stats['reps']

    dist_rhymes, dist_repetitions, stacked_fig = plot_stacked(rhyme, reps, length)

    portion_reps = sum(dist_repetitions.values())
    portion_real = 1 - dist_rhymes['ABCD'] - portion_reps

    quatrain_path = training_data_path + '/stats.pkl'
        
    p_meter = chi2_test(quatrain_path, sample_path, 'meter')
    p_rhyme = chi2_test(quatrain_path, sample_path, 'rhyme')

    res['p_meter'] = float(p_meter)
    res['p_rhyme'] = float(p_rhyme)
    res['real_rhymes'] = float(portion_real)
    res['repetitions'] = float(portion_reps)
    res['rhymes'] = dist_rhymes
    res['reps'] = dist_repetitions

    # plot meters
    meter = {k: v / length for k, v in meter.items()}
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.2)
    plt.style.use('seaborn-whitegrid')
    plt.xticks(rotation=30)
    plt.bar(meter.keys(), meter.values(), color = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.ylabel("Density", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    return res, stacked_fig, fig


def process_training_data(quatrain_path):
    
    # we need the statistics files containing information about meter, rhyme, repetition
    quatrain_path = quatrain_path + '/stats.pkl'
    
    # load stats file
    with open(quatrain_path, 'rb') as f:
        quatrain = pickle.load(f)

    rhyme = quatrain['rhyme']
    reps = quatrain['reps']
    meter = quatrain['meter']
    length = quatrain['length']
    _, _, stacked_fig = plot_stacked(rhyme, reps, length)

    meter = {k: v / length for k, v in meter.items()}
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.2)
    plt.style.use('seaborn-whitegrid')
    plt.xticks(rotation=30)
    plt.bar(meter.keys(), meter.values(), color = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.ylabel("Density", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return stacked_fig, fig

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_only', action="store_true")
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--quatrain_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    if args.save_path:
        # create save path if non existent
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    if args.training_data_only == True:
        stacked_rhymes, meters = process_training_data(args.quatrain_path)
        if args.save_path:
            stacked_rhymes.savefig(args.save_path + '/' + 'rhyme.png', dpi=100)
            meters.savefig(args.save_path + '/' + 'meter.png', dpi=100)
        
    else:
        res, stacked_rhymes, meters = meter_rhyme_metrics(args.sample_path, args.quatrain_path)
        if args.save_path:
            stacked_rhymes.savefig(args.save_path + '/' + 'rhyme.png', dpi=100)
            meters.savefig(args.save_path + '/' + 'meter.png', dpi=100)
        pprint.pprint(res)

    


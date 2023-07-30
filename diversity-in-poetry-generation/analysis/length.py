from datasets import load_from_disk
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import wasserstein_distance
import numpy as np
import os
import json
import argparse

current_path = os.path.dirname(os.path.realpath(__file__))
sample_path = current_path + '/samples/'
length_path=current_path + '/data/length/'


# chi square test
def get_contigents_len(ds1, ds2):
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
    ds1 = load_from_disk(path1)
    ds2 = load_from_disk(path2)
    c = get_contigents_len(ds1, ds2)
    return chi2_contingency(c)[1]


def histogram_intersection(path1, path2):
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
    ds = load_from_disk(path)
    length = ds['length']
    mean = np.mean(length)
    std = np.std(length)
    return mean, std


def min_max(path):
    ds = load_from_disk(path)
    length = ds['length']
    return np.min(length), np.max(length)


def wasserstein(path1, path2):
    ds1 = load_from_disk(path1)
    ds2 = load_from_disk(path2)
    length1 = ds1['length']
    length2 = ds2['length']
    return wasserstein_distance(length1, length2)


def store(model, lang, args):
    
    model_path = current_path + '/samples/' + model + '/' + lang 
    save_path = current_path + '/data/length/' + model + '.txt'
    
    if lang == 'en':
        QuaTrain = current_path + '/data/training_data/QuaTrain/'
    else:
        QuaTrain = current_path + '/data/training_data/QuaTrain-de/'

    for t, p, k, pen in args:
        name = 'temp{}_top_k{}_top_p{}_pen{}'.format(t,k,p,pen)
        sample_path = model_path + '/' + name + '/'
        
        p = chi_square_len(QuaTrain, sample_path)
        
        with open(save_path, "a") as myfile:
            myfile.write('{} : {}\n'.format(name, p))


def plot_len_figures(sample_path, name, lang, save_directory):
    
    ds = load_from_disk(sample_path)
    length = ds['length']
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.style.use('seaborn-whitegrid')
    plt.hist(length, bins='auto', facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, density=True)
    #plt.title(name + ": Max Similarity")
    plt.xlabel("Quatrain Length", fontsize=17)
    plt.ylabel("Densitiy", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(save_directory + '/' + lang + '-hist-' +  name + '.png', dpi=100)


def length_metrics(model, lang, combs):
    
    value_dict = {}
    path = sample_path + model + '/' + lang + '/'
    
    if lang == 'en':
        quatrain_path = current_path + '/data/training_data/QuaTrain'
    else:
        quatrain_path = current_path + '/data/training_data/QuaTrain-de'

    save_directory = length_path + model

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for t, k, p, pen in combs:
        name ='temp{}_top_k{}_top_p{}_pen{}'.format(t, k, p, pen)
        sample = path + name
        
        m, sd = mean_sd(sample)
        minimum, maximum = min_max(sample)

        hist_intersection = histogram_intersection(quatrain_path, sample)
        wasserstein_distance = wasserstein(quatrain_path, sample)
        
        value_dict[name] = {}

        value_dict[name]['mean'] = float(m)
        value_dict[name]['standard_deviation'] = float(sd)
        value_dict[name]['min'] = int(minimum)
        value_dict[name]['max'] = int(maximum)
        value_dict[name]['histogram_intersection'] = float(hist_intersection)
        value_dict[name]['wasserstein_distance'] = float(wasserstein_distance)

        plot_len_figures(sample, name, lang, save_directory)
    
    with open(save_directory + '/' + 'metrics_' + lang + '.json', 'w') as f:
        json.dump(value_dict, f)
 
    return value_dict


def process_training_data(lang):
    
    if lang == 'en':
        quatrain_path = current_path + '/data/training_data/QuaTrain'
    else:
        quatrain_path = current_path + '/data/training_data/QuaTrain-de'
    
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
    fig.savefig(current_path + '/data/length/' + lang + '-Quatrain.png', dpi=100)
    
    with open(current_path + '/data/length/' + lang + '-QuaTrain.json', 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str)
    parser.add_argument('--quatrain_path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--out_path', type=str)

    args = parser.parse_args()

    value_dict = {}

    m, sd = mean_sd(args.sample_path)
    minimum, maximum = min_max(args.sample_path)

    hist_intersection = histogram_intersection(args.quatrain_path, args.sample_path)
    wasserstein_distance = wasserstein(args.quatrain_path, args.sample_path)







    
    




process_training_data('en')
process_training_data('de')



# english and german
models1 = ['gpt2-small', 'gpt2-large', 'poetry-gpt2-small', 'poetry-gpt2-large', 'bygpt5-base', 'bygpt5-medium',
           "poetry-bygpt5-medium", "poetry-bygpt5-base"]

# only english
models2 = ['gptneo-small', 'gptneo-xl', 'poetry-gptneo-small', 'poetry-gptneo-xl']

# german and english
models3 = ['deepspeare']

# only english
models4 = ['structured-adversary']

combs = [(1.0, 0, 1.0, None), #vanilla
        (1.0, 10, 1.0, 0.6), #contrastive
        (1.0, 6, 1.0, 0.7), #constrastive
        (0.7, 0, 0.9, None), #temp, p
        (1.0, 0, 0.9, None), #p
        (0.7, 0, 0.7, None), #temp, p
        (1.0, 0, 0.7, None), #p
        (1.0, 10, 1.0, None), #top k
        (0.7, 10, 1.0, None), #temp topk
        (1.0, 25, 1.0, None), #top k
        (0.7, 25, 1.0, None), #temp topk
        ]

combs2 = [(1.0, 0, 1.0, None),
          (0.7, 0, 1.0, None)]

""" for model in models1:
    values = length_metrics(model, 'en', combs)
    values = length_metrics(model, 'de', combs)

for model in models2:
    values = length_metrics(model, 'en', combs) """

for model in models3:
    values = length_metrics(model, 'en', combs2)
    values = length_metrics(model, 'de', combs2)

#for model in models4:
 #   values = length_metrics(model, 'en', combs2)
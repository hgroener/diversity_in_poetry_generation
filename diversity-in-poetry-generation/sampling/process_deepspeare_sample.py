import sys
import os
import argparse
import pickle

# routine to import parent folder
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from helper_functions import *


def process_ds(path, lang, temp):
    """
    Processes a raw text file containing quatrains generated by Deep-speare to an annotated dataset 
    and derives statistics and distributions about rhyme, meter and length

    Parameters:
    ----------
    path : Path to text file
    lang : Language of the poems. Either English ('en') or German ('de')
    temp : Temperature used during Deep-speare inference

    Returns:
    -------
    quatrains : Processed and annotated dataset
    stats : Dictionary containing different statistics
    dist : Dictionary containing different normalzed distributions
    """
    with open(path) as f:
        lines = f.readlines()
    quatrains = []
    for i in range(2,len(lines)-3,6):
        quatrain = []
        for j in range(i,i+4):
            quatrain.append(lines[j][14:].replace('\n', ''))
        quatrains.append(quatrain)
    quatrains = get_dataset(quatrains, lang)
    quatrains, stats = processQuatrains(quatrains, lang=lang)
    get_fake_rhymes(quatrains, stats)
    dist = get_dist(stats, temp)
    return quatrains, stats, dist


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_sample_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--temperature", type=int)
    args = parser.parse_args()

    # process raw text file to a annotated dataset and derive different statistics and distributions 
    quatrains, stats, dist = process_ds(args.raw_sample_path, args.lang, args.temperature)

    # make sure that the save path is created if non existent
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # save dataset and dictionaries to a path of choice
    quatrains.save_to_disk(args.save_path + '/' + args.save_name)
    
    with open(args.save_path + '/' + args.save_name + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    with open(args.save_path + '/' + args.save_name + '/dist.pkl', 'wb') as f:
        pickle.dump(dist, f)

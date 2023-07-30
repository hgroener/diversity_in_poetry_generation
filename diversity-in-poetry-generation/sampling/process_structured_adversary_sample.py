import sys
import os
import json

# routine to import parent folder
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from helper_functions import *

def process_sa(path, temp, lang='en', num_samples=500):
    with open(path) as f:
        lines = json.load(f)
    quatrains = []
    print('reading done')
    for quatrain in lines[:num_samples]:
        quatrains.append(quatrain[1].split('<eos>'))
    print('adding to list done')
    quatrains = get_dataset(quatrains, lang)
    print('build dataset done')
    quatrains, stats = processQuatrains(quatrains, lang=lang)
    print('processed')
    get_fake_rhymes(quatrains, stats)
    dist = get_dist(stats, temp)
    return quatrains, stats, dist
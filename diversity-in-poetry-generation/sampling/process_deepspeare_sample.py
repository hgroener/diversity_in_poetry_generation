import sys
import os

# routine to import parent folder
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from helper_functions import *

def process_ds(path, lang, temp):
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

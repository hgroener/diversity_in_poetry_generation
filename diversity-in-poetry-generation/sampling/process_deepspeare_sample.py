import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
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

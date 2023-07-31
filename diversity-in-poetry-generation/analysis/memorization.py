from datasets import load_from_disk
import os
import sys
import pprint

# faster implementation of SequenceMatcher
from cydifflib import SequenceMatcher
import difflib
difflib.SequenceMatcher = SequenceMatcher

import argparse

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


def memorization(quatrains, generated_data_path, cutoff, num_to_test):

    generated_data = load_from_disk(generated_data_path)

    if num_to_test:
        generated_data = generated_data.filter(lambda _, idx: idx <= num_to_test-1, with_indices=True)

    generated_data = flatten_list(generated_data.map(join)['text'])
    num_copied = 0
    for quatrain in generated_data:
        if difflib.get_close_matches(quatrain, quatrains, n=1, cutoff=cutoff):
            num_copied += 1
    return num_copied / len(generated_data)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--quatrain_path", type=str)
    parser.add_argument("--cutoff", type=float, default=0.7)
    parser.add_argument("--num_to_test", type=int, default=None)
    args = parser.parse_args()


    quatrains = load_from_disk(args.quatrain_path)
    quatrains = flatten_list(quatrains.map(join)['text'])

    memorization_rate = memorization(quatrains=quatrains, generated_data=args.sample_path, 
                                    cutoff=args.cutoff, num_to_test=args.num_to_test)

    pprint.pprint(memorization_rate)
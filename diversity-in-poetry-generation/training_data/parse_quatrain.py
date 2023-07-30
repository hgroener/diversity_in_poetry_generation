from uniformers.datasets import load_dataset
import os
import argparse
import sys
import pickle

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str)
    args = parser.parse_args()

    output_dir="quatrain_data/" + args.lang 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    quatrains = load_dataset('quatrain', lang=args.lang, split="train")
    
    # annotate dataset and derive basic statistics
    quatrains, stats = processQuatrains(quatrains, args.lang)
    get_fake_rhymes(quatrains, stats)
    
    # save processed dataset and statistics
    quatrains.save_to_disk(output_dir + '/QuaTrain_' + args.lang)
    with open(output_dir + '/QuaTrain_' + args.lang + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    # for unconditioned training, save a text file containing raw quatrains
    quatrains = quatrains['text']

    with open(output_dir + '/QuaTrain.txt', 'w') as f:
        for quatrain in quatrains:
            f.write('\n'.join(quatrain))
            f.write('\n\n')


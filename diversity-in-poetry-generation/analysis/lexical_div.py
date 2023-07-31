import os
from lexical_diversity import lex_div as ld
from datasets import load_from_disk
import sys
import argparse
import pprint

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


def tokenize(quatrains):
    tokens = []
    flat = flatten_list(quatrains.map(join)['text'])
    for quatrain in flat:
        tokens.append(ld.flemmatize(quatrain))
    # returns tokenized sentences
    return tokens


def averaged_ttr(sentence_tokens):
    ttr = 0
    for sentence in sentence_tokens:
        ttr += ld.ttr(sentence)
    return ttr/len(sentence_tokens) 
    

def lexical_div(quatrains):
    res = {}
    sentence_tokens = tokenize(quatrains)
    #tokenized corpus
    tokens = flatten_list(sentence_tokens)
    avg_ttr = averaged_ttr(sentence_tokens)
    mattr = ld.mattr(tokens, window_length=35)
    mtld = ld.mtld(tokens)
    hdd = ld.hdd(tokens)
    res['averaged_ttr'] = avg_ttr
    res['mattr'] = mattr
    res['mtld'] = mtld
    res['hdd'] = hdd
    return res


def ld_quatrain(training_data_path):
    
    quatrain = load_from_disk(training_data_path)
    
    res = lexical_div(quatrain)

    return res


def ld_metrics(sample_path):
    
    loaded_sample = load_from_disk(sample_path)

    res = lexical_div(loaded_sample)
    
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_only', action="store_true")
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--quatrain_path", type=str)
    args = parser.parse_args()

    if args.training_data_only == True:
        res = ld_quatrain(args.quatrain_path)
    else:
        res = ld_metrics(args.sample_path)
    
    pprint.pprint(res)


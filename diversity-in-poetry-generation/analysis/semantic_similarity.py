from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
from scipy.stats import normaltest
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import sys
import argparse
import pprint

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *

#emb = embedded Quatrain
# sentences = samples
# load samples in function
# load emb also every step?

def max_sim_dist(emb, sample_path, model='distiluse-base-multilingual-cased-v1', device='cuda:0'):
    model = SentenceTransformer(model)
    sentences = load_from_disk(sample_path)
    sentences = flatten_list(sentences.map(join)['text'])
    res = []
    for sentence in sentences:
        sentence_embedding = model.encode(sentence, convert_to_tensor=True, device=device)
        top_k=1
        cos_scores = util.pytorch_cos_sim(sentence_embedding, emb)[0].cpu()
        top_result = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        for idx in top_result[0:top_k]:
            res.append(cos_scores[idx])
    return res


def ss_metrics(sample_path, emb_path):

    #load pretrained corpus embeddings
    emb = torch.load(emb_path)

    sims = max_sim_dist(emb, sample_path)

    mean_sim = np.mean(sims)
    sd_sim = np.std(sims)

    minimum = np.min(sims)
    maximum = np.max(sims)
        
    sims = np.array(sims)

    p = normaltest(sims).pvalue
        
    res = {}

    res['mean'] = float(mean_sim)
    res['standard_deviation'] = float(sd_sim)
    res['min'] = float(minimum)
    res['max'] = float(maximum)
    res['dagostino_p'] = float(p)

    # plot similarity values    
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.style.use('seaborn-whitegrid')
    plt.hist(sims, bins=30, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    #plt.title(name + ": Max Similarity")
    plt.xlabel("Cosine Similarity", fontsize=17)
    plt.ylabel("Frequency", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    return res, fig

    
def create_corpus_embeddings(training_data_path, save_path, lang):

    sentences = load_from_disk(training_data_path)
    sentences = flatten_list(sentences.map(join)['text'])

    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(sentences, convert_to_tensor=True)

    torch.save(embeddings, save_path + '/' + lang + '-QuaTrain.pt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_only', action="store_true")
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--emb_path", type=str)
    parser.add_argument("--quatrain_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()

    # create initial corpus embeddings
    if args.training_data_only == True:
        create_corpus_embeddings(args.quatrain_path, args.save_path, args.lang)
    
    else:
        metrics, similarity_figure = ss_metrics(args.sample_path, args.emb_path)
        if args.save_path:
            similarity_figure.savefig(args.save_path + '/similarities.png')
        pprint.pprint(metrics)
        
        


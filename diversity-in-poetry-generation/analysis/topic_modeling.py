from datasets import load_from_disk
import os
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from matplotlib import pyplot as plt
from collections import Counter
import argparse
from math import log10, floor
import random
import sys
import pprint
import pickle


# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *

from train_topic_model import flemmatize


def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def get_dominant_topics(sample_path, trained_topic_model, id2word, lang):
    samples = load_from_disk(sample_path)
    
    if len(samples) > 500:
        samples = samples.filter(lambda _, idx: idx <= 499, with_indices=True)
    
    samples = flemmatize(flatten_list(flatten_list(samples.map(join)['text'])), lang)
    samples_corpus = [id2word.doc2bow(text) for text in samples]
    
    samples_topic_vectors = []
    samples_topic_vectors_distilled = []
    for quatrain in samples_corpus:
        samples_topic_vectors.append(trained_topic_model.get_document_topics(quatrain, minimum_probability=0))
        samples_topic_vectors_distilled.append(trained_topic_model.get_document_topics(quatrain))
    most_probable_topics = []
    for quatrain in samples_topic_vectors_distilled:
        quatrain.sort(key = lambda x: x[1], reverse=True, )
        if len(quatrain) > 0:
            #tup = random.choice(quatrain)
            most_probable_topics.append(quatrain[0][0])
            #most_probable_topics.append(tup[0])
    
    num_topics = []
    for entry in samples_topic_vectors_distilled:
        for tup in entry:
            num_topics.append(tup[0])
    num_topics = list(set(num_topics))
    
    return samples_topic_vectors, most_probable_topics, len(num_topics)


def tm_metrics(sample_path, trained_model_path, lang):
    
    value_dict = {}
    
    LDA = LdaMulticore.load(trained_model_path + 'model')
    id2word = corpora.Dictionary.load(trained_model_path + 'model.id2word')
    num_topics_quatrain = LDA.num_topics

    _, tops, num_topics_in_sample = get_dominant_topics(sample_path, LDA, id2word, lang)

    c = Counter(tops)
    top10 = dict(c.most_common(10))
    top10_topics = list(top10.keys())
        
    value_dict['percentage'] = num_topics_in_sample/num_topics_quatrain

    for i, topic in enumerate(top10_topics):
        value_dict['top_{}'.format(i+1)] = str([i[0] for i in LDA.show_topic(topic, topn=10)])
    
    # create figure
    fig = plt.figure(figsize=(5,4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.style.use('seaborn-whitegrid')
    topics = [str(i) for i in list(top10.keys())]
    plt.barh(topics, list(top10.values()), align='center',  color='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.xlabel("Frequency", fontsize=17)
    plt.ylabel("Topic ID", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().invert_yaxis()

    return value_dict, fig

def most_relevant_topics_quatrain(trained_path):
    
    LDA = LdaMulticore.load(trained_path + 'model')
    corpus = corpora.MmCorpus(trained_path + 'corpus.mm')

    samples_topic_vectors = []

    for quatrain in corpus:
        samples_topic_vectors.append(LDA.get_document_topics(quatrain, minimum_probability=0))
    
    most_probable_topics = []
    for quatrain in samples_topic_vectors:
        quatrain.sort(key = lambda x: x[1], reverse=True, )
        if len(quatrain) > 0:
            most_probable_topics.append(quatrain[0][0])
    
    c = Counter(most_probable_topics)
    top10 = dict(c.most_common(10))
    top10_topics = list(top10.keys())

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(5,4))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    topics = [str(i) for i in top10_topics]
    plt.barh(topics, list(top10.values()), align='center',  color='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.xlabel("Frequency", fontsize=17)
    plt.ylabel("Topic ID", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().invert_yaxis()

    values = {}
    for i, topic in enumerate(top10_topics):
            values['top_{}'.format(i+1)] = str([i[0] for i in LDA.show_topic(topic, topn=10)])
    
    return values, fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_only', action="store_true")
    parser.add_argument("--trained_model_path", type=str)
    parser.add_argument("--sample_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    # load language specific predefined stop words
    with open(current_path + "/stop_en", "rb") as f:
        stop_en = pickle.load(f)
    
    with open(current_path + "/stop_de", "rb") as f:
        stop_de = pickle.load(f)
    
    if args.save_path:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    if args.training_data_only == True:
       top_topics, top_bar = most_relevant_topics_quatrain(args.trained_model_path)
       if args.save_path:
           top_bar.savefig(args.save_path + '/' + args.lang + '-top10-QuaTrain.png', dpi=100)
       pprint.pprint(top_topics)
    else:
        top_topics, top_bar = tm_metrics(args.sample_path, args.trained_model_path, args.sample_path)
        if args.save_path:
           top_bar.savefig(args.save_path + '/' + args.lang + '-top10.png', dpi=100)
        pprint.pprint(top_topics)


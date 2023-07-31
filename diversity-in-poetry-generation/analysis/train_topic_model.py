import sys
import string
from datasets import load_from_disk
from nltk.corpus import stopwords
import pickle

# Gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


from matplotlib import pyplot as plt
import os
import argparse


# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


def flemmatize(flat_corpus, lang):
    """
    Lemmatizes a corpus of quatrains, removes predefined stop words, removes
    puncuation and filteres out short tokens
    
    Parameters:
    ----------
    flat_corpus : Corpus of quatrains that has been flattened (e.g. the inner list
    structure is removed)
    lang : language 

    Returns:
    -------
    res : list of tokens
    """
    punct = string.punctuation + '“'
    punct = punct + '”'
    punct = punct + '’'
    punct = punct + '‘'
    punct = punct + '—'
    if lang == 'en':
        nlp = spacy.load('en_core_web_sm')
        stopWords_spacy = list(nlp.Defaults.stop_words)
        stopWords = stopwords.words('english')
        stopWords = stopWords + stop_en + stopWords_spacy
        stopWords.extend(['', 'thy', 'thee', 'thine', 'thyself', 'hath', 'leander', 'doth', 'oer', "'nt",
                          'pour', 'sans', 'quon', 'mais', 'bien', 'nous', 'homme', 'peut', 'quatre', 'tient',
                          'ducats', 'cinq', 'raison', 'tuer', 'pomme', 'fêtu', 'malles', 'ille', 'cerebri', 'fuit', 
                          'sheridan', 'pallas', 'quae', 'deorum', 'quisnam', 'melior', 'orto', 'artem', 'tibi', 
                          'natura', 'atque', 'nascenti', 'cunabula', 'scrutandi', 'genium', 'rimandi', 'tradidit', 
                          'puerorum', 'besieger', 'mayë', 'hallgerd', 'whitewater', 'ingathere', 'daïs', 'snæbiorn', 
                          'howso', 'graithe', 'hallbiorn'])
    else:
        nlp = spacy.load('de_core_news_sm')
        stopWords = stopwords.words('german')
        stopWords_spacy = list(nlp.Defaults.stop_words)
        stopWords = stopWords + stop_de + stopWords_spacy
        stopWords.extend(['vnd', 'kan', 'dieß', 'laß', 'ward', 'bey', 'diß', 'vör', 'all', 'itzt',
                         'voll', 'dat', 'seyn', 'stets', 'ick', 'auff', 'sieht', 'hertz', 'hefftig',
                         'ums', 'manch', 'rief', 'ains', 'gütt', 'fast', 'lassen', 'drauf', 'hört',
                         'gutem', 'heißt', 'liegt', 'äußre', 'stand', 'tho', 'hält', 'offt', 'bistu',
                         'vorhin', 'hätt', 'sey', 'hei', 'komm', 'wol', 'sprach', 'lie', 'unsrer',
                         'guten', 'durchs', 'ans', 'wär', 'nich', 'ists'])
    stopWords = set(stopWords)
    res = []
    for entry in flat_corpus:
        doc = nlp(entry)
        lemmas = [token.lemma_ for token in doc]
        a_lemmas = [lemma.lower() for lemma in lemmas if lemma.isalpha()]
        wordsFiltered = []
        for w in a_lemmas:
            if w not in stopWords:
                if w not in punct:
                    if len(w) > 3:
                        wordsFiltered.append(w)
        res.append(wordsFiltered)
    return res


def compute_coherence_values(dictionary, corpus, texts, limit, start, step, workers):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    start : Min num of topics
    step : Step size used to increase the num of topics
    workers : Number of CPU threads

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(workers=workers,corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        perp = model.log_perplexity(corpus)
        coherence = coherencemodel.get_coherence()
        coherence_values.append(coherence)
        print('{} processed with coherence value {} and perplexity {}'.format(num_topics, coherence, perp))

    return model_list, coherence_values


def LDA(ds, limit, minTopics, maxTopics, step, workers, lang):
    """
    Determine the optimal number of topics based on the c_v score

    Parameters:
    ----------
    ds : Training dataset
    limit : Debug parameter to limit the number of quatrains to be processed by the model
    min_topics : Min num of topics
    max_topics : Max num of topics
    step : Step size used to increase the num of topics
    workers : Number of CPU threads
    lang : language

    Returns: 
    -------
    optimal_model : LDA model achieving the highest c_v score
    corpus : Doc2bow representation of the training corpus
    id2word : Dictionary with word/id pairs
    fig : Plot of coherence values as a function of number of topics
    """
    # Flatten and lemmatize dataset 
    if limit and limit < len(ds):
        ds = ds.shuffle().filter(lambda _, idx: idx <= limit - 1, with_indices=True)
    
    ds = flemmatize(flatten_list(flatten_list(ds.map(join)['text'])), lang=lang)
    
    # Create Dictionary
    id2word = corpora.Dictionary(ds)

    # Create Corpus
    texts = ds

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, 
                                                            start=minTopics, limit=maxTopics, step=step, 
                                                            workers=workers)
    # Show graph
    limit=maxTopics; start=minTopics; step=step;
    x = range(start, limit, step)
    fig = plt.figure(figsize=(5,4))
    plt.style.use('seaborn-whitegrid')
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.2)
    plt.plot(x, coherence_values)
    plt.xlabel("Topics", fontsize=17)
    plt.ylabel("Coherence score", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    max_coherence = max(coherence_values)
    best_result_index = coherence_values.index(max_coherence)
    optimal_model = model_list[best_result_index]
    num_topics = optimal_model.get_topics().shape[0]
    print('Optimal model has %d topics with coherence score %s and perplexity %s'
          % (num_topics, max_coherence, optimal_model.log_perplexity(corpus)))
    
    return optimal_model, corpus, id2word, fig


def visualizeTopics(m, c, ids, mds='mmds'):
    """
    Generates a pyldavis object representing a visualization of the topic model
    
    Parameters:
    ----------
    m : LDA model
    c : Doc2bow corpus
    ids : Dictionary of word/id pairs

    Returns:
    -------
    vis : pyldavis object
    """
    vis = pyLDAvis.gensim_models.prepare(m, c, ids, mds=mds)
    return vis


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quatrain_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min_topics', type=int, default=50)
    parser.add_argument('--max_topics', type=int, default=250)
    parser.add_argument('--stepsize', type=int, default=10)
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # load language specific predefined stop words
    with open(current_path + "/stop_en", "rb") as f:
        stop_en = pickle.load(f)
    
    with open(current_path + "/stop_de", "rb") as f:
        stop_de = pickle.load(f)
     
    QuaTrain = load_from_disk(args.quatrain_path)

    m, c, ids, fig = LDA(QuaTrain, minTopics=args.min_topics, maxTopics=args.max_topics, limit=args.limit, 
                step=args.stepsize, lang=args.lang, workers=args.workers)
    
    fig.savefig(args.save_path + '/coherence.png')
    
    m.save(args.save_path + '/model')
    corpora.MmCorpus.serialize(args.save_path + '/corpus.mm', c)
    
    vis = visualizeTopics(m, c, ids, mds='mmds')
    pyLDAvis.save_json(vis, args.save_path + '/vis/vis.json')
    pyLDAvis.save_html(vis, args.save_path + '/vis/vis.html') 
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers import pipeline
import pickle
import string
import random
import torch
import os
import sys

from transformers import logging
logging.set_verbosity_warning()

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *

def generate_quatrains(model, lang='en', min_length=20, max_length=100, num_samples=1000, save=False, device=0,
                      temperature=1.0, top_k=0, top_p = 1.0, total=100, penalty_alpha=None, do_sample=True,
                      num_beams=1, num_beam_groups=1):
    
    if 'bygpt5' in model:
        generator = TextGenerationPipeline(model=ByGPT5LMHeadModel.from_pretrained(model),
                                           tokenizer=ByGPT5Tokenizer.from_pretrained(model),
                                           device=device)
        prefix=ByGPT5Tokenizer.from_pretrained(model).eos_token
    else:
        generator = pipeline('text-generation', model=model, device=device)
        prefix='<|endoftext|>'
    
    samples=[]
    for i in range(total):
        gen = generator("", do_sample=do_sample, min_length = min_length, 
                        max_length=max_length, prefix=prefix, 
                        num_return_sequences=num_samples, 
                        temperature=temperature, top_k=top_k, top_p=top_p, 
                        penalty_alpha=penalty_alpha, num_beams=num_beams,
                        num_beam_groups=num_beam_groups)
        quatrains = []
        for sample in gen:
            for quatrain in sample['generated_text'].split('\n\n'):
                quatrains.append(quatrain.split('\n'))
        quatrains = [quatrain for quatrain in quatrains if len(quatrain) == 4]
        for quatrain in quatrains:
            for line in quatrain:
                if len(line) == 0:
                    quatrains.remove(quatrain)
                    break
        if len(quatrains) == 0:
            continue
        for quatrain in quatrains:
            samples.append(quatrain)
        #samples.append(random.choice(quatrains))
        
    samples = get_dataset(samples, lang)
    del generator
    torch.cuda.empty_cache()

    return samples


def sample_pipeline(model, total=100, temperature=1.0, top_k=0, top_p=1.0, penalty_alpha=None, lang='en',
                    do_sample=True, num_beams=1, num_beam_groups=1, num_samples=5, min_length=50,
                    max_length=100):
    
    sample = generate_quatrains(model=model, num_samples=num_samples, total=total, 
                                temperature=temperature, top_k=top_k, 
                                top_p=top_p, penalty_alpha=penalty_alpha,
                                do_sample=do_sample,num_beams=num_beams,
                                num_beam_groups=num_beam_groups, lang=lang,
                                max_length=max_length, min_length=min_length)
    
    torch.cuda.empty_cache()

    sample, stats = processQuatrains(sample,lang)
    get_fake_rhymes(sample, stats)
    dist = get_dist(stats, temp=temperature, top_k=top_k, top_p=top_p, num_beams=num_beams, 
                    num_beam_groups=num_beam_groups, do_sample=do_sample, penalty_alpha=penalty_alpha)
    
    return sample, stats, dist


def get_last_word(quatrain):
    res = []
    for line in quatrain:
        line = line.translate(str.maketrans('', '', string.punctuation))
        if len(line) == 0:
            continue
        split = line.split()
        if len(split) > 0:
            res.append(line.split()[-1])
    return res


def store_samples(model, lang='en', total=500, num_samples=10, temperature=1.0, top_p=1.0, top_k=0, 
                  penalty_alpha=None, max_length=100, min_length=50, local=True):
    
    if local:
        model_path = current_path + '/models/experiments/' + model + '/' + lang
        save_path = current_path + '/samples/' + model + '/' + lang
    else:
        model_path = model
        save_path = current_path + '/samples/' + model.split('/')[1][:-3] + '/' + lang

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sample, stats, dist = sample_pipeline(model_path, total=total, lang=lang, temperature=temperature,
                                          top_k=top_k, top_p=top_p, penalty_alpha=penalty_alpha, 
                                          num_samples=num_samples, max_length=max_length,
                                          min_length=min_length)


    #save_path = current_path + '/samples/' + model + '/' + lang


    name ='temp{}_top_k{}_top_p{}_pen{}'.format(temperature, top_k, top_p, penalty_alpha)

    sample.save_to_disk(save_path + '/' + name)
    with open(save_path + '/' + name + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    with open(save_path + '/' + name + '/dist.pkl', 'wb') as f:
        pickle.dump(dist, f)

def call_store(models, langs, min, max, local, args, total=10, num_samples=50):
    for model in models:
        for lang in langs:
            for temp, p, k, pen in args:
                store_samples(model, lang, total, num_samples, temp, p, k, pen, max, min, local)
                print('{} {} done with temp {}, top_p {}, top_k {}, penalty_alpha {}'.format(model, lang, temp, p, k, pen))
    
#for model in MODELS:
  #  store_samples(model)

#for model in MODELS_DE:
 #   store_samples(model, lang='de', total=200)

#model='debug/poetry-gpt2-small-tes/en'
#store_samples(model, lang='en', total=100)

#print('yo')

#start = time.time()
#model = current_path + '/models/experiments/bygpt5-medium/en'
#model = "nllg/poetry-bygpt5-medium-en"
#sample, stats, dist = sample_pipeline(model, total=10, num_samples=10, lang='en', min_length=50, max_length=350, top_k=10, 
#                                      penalty_alpha=0.6)
#print(sample[0])
#print('\n')
#print(stats)
#print('\n')
#print(dist)
#print('\n')
#print(np.min(sample['length']))
#print(np.mean(sample['length']))
#print(np.max(sample['length']))
#end = time.time()
#print(end - start)



# english and german, min=40, max=150
models1 = ['gpt2-small', 'gpt2-large', 'poetry-gpt2-small', 'poetry-gpt2-large']
#models1 = ['poetry-gpt2-large']

# only english, min=40, max=150
models2 = ['gptneo-small', 'gptneo-xl', 'poetry-gptneo-small', 'poetry-gptneo-xl']

# english and german, min=50, max=300
models3 = ['bygpt5-base', 'bygpt5-medium']

# english, non local, min=50, max=300
models4 = ["nllg/poetry-bygpt5-medium-en", "nllg/poetry-bygpt5-base-en"]

# german, non local, min=50, max=300
models5 = ["nllg/poetry-bygpt5-medium-de", "nllg/poetry-bygpt5-base-de"]

# temp, p, k, pen, max, min, local
args = [(1.0, 1.0, 0, None), #vanilla
        (1.0, 1.0, 10, 0.6), #contrastive
        (1.0, 1.0, 6, 0.7), #constrastive
        (0.7, 0.9, 0, None), #temp, p
        (1.0, 0.9, 0, None), #p
        (0.7, 0.7, 0, None), #temp, p
        (1.0, 0.7, 0, None), #p
        (1.0, 1.0, 10, None), #top k
        (0.7, 1.0, 10, None), #temp topk
        (1.0, 1.0, 25, None), #top k
        (0.7, 1.0, 25, None), #temp topk
        ]

#args = [(0.7, 0.7, 0, None), (1.0, 0.7, 0, None)]
#args = [(1.0, 0.7, 0, None)]

#call_store(models1, ['de'], 40, 120, True, args, 10, 50)
#call_store(models2, ['en'], 40, 150, True, args, 10, 50)
#call_store(models3, ['en', 'de'], 50, 300, True, args, 10, 50)
call_store(models4, ['en'], 0, 350, False, args, 10, 50)
call_store(models5, ['de'], 0, 350, False, args, 10, 50)


#store_samples(model="poetry-gpt2-small", lang='en', total=10, num_samples=100, min_length=40, max_length=150)
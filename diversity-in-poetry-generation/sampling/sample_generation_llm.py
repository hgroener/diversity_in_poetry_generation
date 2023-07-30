from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers import pipeline
from transformers import logging
import argparse
import pickle
import os
import sys


logging.set_verbosity_warning()

# routine to import parent folder
current_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_path)
sys.path.append(parent)

from helper_functions import *


def generate_quatrains(model, lang, min_length, max_length, num_samples, temperature, top_k,
                       top_p, total, penalty_alpha, do_sample, num_beams, num_beam_groups, device=0):
    
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
        #samples.append(random.choice(quatrains)) TODO: better select randomly
        
    samples = get_dataset(samples, lang)
    return samples


def sample_pipeline(model, lang, total, temperature, top_k, top_p, penalty_alpha, do_sample, num_beams,
                    num_beam_groups, num_samples, min_length, max_length):
    
    sample = generate_quatrains(model=model, num_samples=num_samples, total=total, 
                                temperature=temperature, top_k=top_k, 
                                top_p=top_p, penalty_alpha=penalty_alpha,
                                do_sample=do_sample,num_beams=num_beams,
                                num_beam_groups=num_beam_groups, lang=lang,
                                max_length=max_length, min_length=min_length)
    
    sample, stats = processQuatrains(sample,lang)
    get_fake_rhymes(sample, stats)
    dist = get_dist(stats, temp=temperature, top_k=top_k, top_p=top_p, num_beams=num_beams, 
                    num_beam_groups=num_beam_groups, do_sample=do_sample, penalty_alpha=penalty_alpha)
    
    return sample, stats, dist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str)
    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--do_sample', action="store_false")
    parser.add_argument('--min_length', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--total', type=int, default=500)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--penalty_alpha', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)

    args = parser.parse_args()

    # generate and process samples
    sample, stats, dist = sample_pipeline(model=args.trained_model_path,
                                          lang=args.lang, 
                                          total=args.total,
                                          temperature=args.temperature,
                                          top_k=args.top_k,
                                          top_p=args.top_p,
                                          penalty_alpha=args.penalty_alpha,
                                          do_sample=args.do_sample,
                                          num_beams=args.num_beams,
                                          num_beam_groups=args.num_beam_groups,
                                          num_samples=args.num_samples,
                                          min_length=args.min_length,
                                          max_length=args.max_length,
                                          )
    
    # define name of sample
    name ='temp{}_top_k{}_top_p{}_pen{}'.format(args.temperature, args.top_k, args.top_p, args.penalty_alpha)
    
    # construct save path
    save_path = args.out_path + '/' +  args.model_name + '/' + args.lang

    # create save path if non existent
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save data
    sample.save_to_disk(save_path + '/' + name)

    with open(save_path + '/' + name + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    with open(save_path + '/' + name + '/dist.pkl', 'wb') as f:
        pickle.dump(dist, f)
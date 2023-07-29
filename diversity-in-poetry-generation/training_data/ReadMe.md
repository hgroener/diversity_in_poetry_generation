# Parse Training Data

### Deep-speare and Structured-Adversary

In order to train Deep-speare and Structured-Adversary, we need to provide train, validation and test splits in
sonnet format as well as a set of word vectors serving as background data. To generate sonnet splits from QuaTrain 
and deduce background data via word2vec we execute

```
python parse_sonnets.py --lang en --get_wv True --vector_size 100
```
Refer to *parse_sonnets.py* to get a full list of word2vec parameters. 

### Unconditioned training for Large Language Models (LLMs)

Execute
```
python parse_quatrain.py --lang de
```
to generate text files containing line separated quatrains extracted from QuaTrain. We will use these files as input data 
for training LLMs (e.g. GPT2 and ByGPT5) in an unconditioned manner.

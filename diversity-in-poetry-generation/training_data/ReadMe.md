# Parse Training Data

In order to train Deep-speare and Structured-Adversary, we need to provide train, validation and test splits in
sonnet format as well as a set of word vectors serving as background data. To generate splits and deduce background
data via word2vec we execute

```
python parse_sonnets.py --lang en --get_wv True --vector_size 100
```
Refer to *parse_sonnets.py* to get a full list of word2vec parameters. 

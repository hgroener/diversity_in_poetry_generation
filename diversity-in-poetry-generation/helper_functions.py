from datasets import Dataset
from uniformers.utils import METERS, QUATRAIN_RHYME_SCHEMES, QuatrainProcessing
import string


def flatten_list(_2d_list):
    """
    Flattens a list of lists by removing the inner list structure  

    Parameters:
    ----------
    _2d_list : Two-dimensional list

    Returns:
    -------
    flat_list : Flattened list 
    """
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def join(examples):
    examples['text'] = ' '.join(examples['text'])
    return examples

def get_dataset(quatrains, lang, temps=None):
    ds = {}
    lang_col = [lang] * len(quatrains)
    ds['text'] = quatrains
    ds['language'] = lang_col
    if temps:
        ds['temps'] = temps
    return Dataset.from_dict(ds)


def get_len(entry):
    length = 0
    for line in entry['text']:
        length += len(line.split())
    entry['length'] = length
    return entry


def processQuatrains(quatrains_ds, lang='en', 
                     meter_model='nllg/clf-canine-m', 
                     rhyme_model='nllg/clf-canine-r', 
                     batch_size=256,
                     num_proc=1):
    
    quatrains = quatrains_ds.map(
    QuatrainProcessing(
        lang=lang,
        meter_model_name=meter_model,
        rhyme_model_name=rhyme_model,
        batch_size=batch_size,
    ),
    batched=True, num_proc=num_proc)

    quatrains = quatrains.map(get_len)
    
    stats = {"length": len(quatrains), "meter": {}, "rhyme": {}}
    for scheme in QUATRAIN_RHYME_SCHEMES:
        stats['rhyme'][scheme] = len(quatrains.filter(lambda example: example["rhyme"] == scheme))
    for meter in METERS:
        stats['meter'][meter] = len(quatrains.filter(lambda example: example["meter"] == meter))

    scores, medium, high = quatrains['alliteration'], 0.05, 0.1
    stats['alliteration'] = {
        "low": len(list(filter(lambda x: x < medium, scores)))/len(scores),
        "medium": len(list(filter(lambda x: medium <= x < high, scores)))/len(scores),
        "high": len(list(filter(lambda x: high <= x, scores)))/len(scores)
    }
    
    return quatrains, stats


def get_dist(ds, temp, top_k=None, top_p=None, num_beams=None, do_sample=True, num_beam_groups=None, penalty_alpha=None, keys = ['meter', 'rhyme']):
    dist = {}
    dist['reps'] = {}
    total = ds['length']
    for key in keys:
        values = ds[key]
        dist[key] = {key: round(value / total,4) for key, value in values.items()}
    for scheme in QUATRAIN_RHYME_SCHEMES:
        if ds['reps'][scheme] == 0:
            dist['reps'][scheme] = 0
        else:
            dist['reps'][scheme] = round(ds['reps'][scheme]/ds['rhyme'][scheme], 4)
    dist['temperature'] = temp
    dist['top_k'] = top_k
    dist['top_p'] = top_p
    dist['num_beams'] = num_beams
    dist['num_beam_groups'] = num_beam_groups
    dist['penalty_alpha'] = penalty_alpha
    dist['do_sample'] = do_sample
    return dist


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


def get_fake_rhymes(ds, stats):
    stats['reps'] = {}
    for scheme in QUATRAIN_RHYME_SCHEMES:
        num_fakes=0
        ds_f = ds.filter(lambda example: example["rhyme"] == scheme)
        for entry in ds_f:
            if entry['rhyme'] == scheme:
                last = get_last_word(entry['text'])
                if len(last) != len(set(last)):
                    num_fakes+=1
        stats['reps'][scheme] = num_fakes
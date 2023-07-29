


def process_ds(path, lang, temp):
    with open(path) as f:
        lines = f.readlines()
    quatrains = []
    for i in range(2,len(lines)-3,6):
        quatrain = []
        for j in range(i,i+4):
            quatrain.append(lines[j][14:].replace('\n', ''))
        quatrains.append(quatrain)
    quatrains = get_dataset(quatrains, lang)
    quatrains, stats = processQuatrains(quatrains, lang=lang)
    get_fake_rhymes(quatrains, stats)
    dist = get_dist(stats, temp)
    return quatrains, stats, dist

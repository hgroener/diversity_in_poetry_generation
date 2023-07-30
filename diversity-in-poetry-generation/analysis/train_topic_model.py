import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
import sys

import string

from datasets import load_from_disk
from nltk.corpus import stopwords

# Gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import LdaMulticore

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this #import pyLDAvis.gensim
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


stop_en = ["'ll","'tis","'twas","'ve","10","39","a","a's","able","ableabout","about","above","abroad","abst","accordance","according","accordingly","across","act","actually","ad","added","adj","adopted","ae","af","affected","affecting","affects","after","afterwards","ag","again","against","ago","ah","ahead","ai","ain't","aint","al","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","amoungst","amount","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","ao","apart","apparently","appear","appreciate","appropriate","approximately","aq","ar","are","area","areas","aren","aren't","arent","arise","around","arpa","as","aside","ask","asked","asking","asks","associated","at","au","auth","available","aw","away","awfully","az","b","ba","back","backed","backing","backs","backward","backwards","bb","bd","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","bf","bg","bh","bi","big","bill","billion","biol","bj","bm","bn","bo","both","bottom","br","brief","briefly","bs","bt","but","buy","bv","bw","by","bz","c","c'mon","c's","ca","call","came","can","can't","cannot","cant","caption","case","cases","cause","causes","cc","cd","certain","certainly","cf","cg","ch","changes","ci","ck","cl","clear","clearly","click","cm","cmon","cn","co","co.","com","come","comes","computer","con","concerning","consequently","consider","considering","contain","containing","contains","copy","corresponding","could","could've","couldn","couldn't","couldnt","course","cr","cry","cs","cu","currently","cv","cx","cy","cz","d","dare","daren't","darent","date","de","dear","definitely","describe","described","despite","detail","did","didn","didn't","didnt","differ","different","differently","directly","dj","dk","dm","do","does","doesn","doesn't","doesnt","doing","don","don't","done","dont","doubtful","down","downed","downing","downs","downwards","due","during","dz","e","each","early","ec","ed","edu","ee","effect","eg","eh","eight","eighty","either","eleven","else","elsewhere","empty","end","ended","ending","ends","enough","entirely","er","es","especially","et","et-al","etc","even","evenly","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","fairly","far","farther","felt","few","fewer","ff","fi","fifteen","fifth","fifty","fify","fill","find","finds","fire","first","five","fix","fj","fk","fm","fo","followed","following","follows","for","forever","former","formerly","forth","forty","forward","found","four","fr","free","from","front","full","fully","further","furthered","furthering","furthermore","furthers","fx","g","ga","gave","gb","gd","ge","general","generally","get","gets","getting","gf","gg","gh","gi","give","given","gives","giving","gl","gm","gmt","gn","go","goes","going","gone","good","goods","got","gotten","gov","gp","gq","gr","great","greater","greatest","greetings","group","grouped","grouping","groups","gs","gt","gu","gw","gy","h","had","hadn't","hadnt","half","happens","hardly","has","hasn","hasn't","hasnt","have","haven","haven't","havent","having","he","he'd","he'll","he's","hed","hell","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","herse”","hes","hi","hid","high","higher","highest","him","himself","himse”","his","hither","hk","hm","hn","home","homepage","hopefully","how","how'd","how'll","how's","howbeit","however","hr","ht","htm","html","http","hu","hundred","i","i'd","i'll","i'm","i've","i.e.","id","ie","if","ignored","ii","il","ill","im","immediate","immediately","importance","important","in","inasmuch","inc","inc.","indeed","index","indicate","indicated","indicates","information","inner","inside","insofar","instead","int","interest","interested","interesting","interests","into","invention","inward","io","iq","ir","is","isn","isn't","isnt","it","it'd","it'll","it's","itd","itll","its","itself","itse”","ive","j","je","jm","jo","join","jp","just","k","ke","keep","keeps","kept","keys","kg","kh","ki","kind","km","kn","knew","know","known","knows","kp","kr","kw","ky","kz","l","la","large","largely","last","lately","later","latest","latter","latterly","lb","lc","least","length","less","lest","let","let's","lets","li","like","liked","likely","likewise","line","little","lk","ll","long","longer","longest","look","looking","looks","low","lower","lr","ls","lt","ltd","lu","lv","ly","m","ma","made","mainly","make","makes","making","man","many","may","maybe","mayn't","maynt","mc","md","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","mh","microsoft","might","might've","mightn't","mightnt","mil","mill","million","mine","minus","miss","mk","ml","mm","mn","mo","more","moreover","most","mostly","move","mp","mq","mr","mrs","ms","msie","mt","mu","much","mug","must","must've","mustn't","mustnt","mv","mw","mx","my","myself","myse”","mz","n","na","name","namely","nay","nc","nd","ne","near","nearly","necessarily","necessary","need","needed","needing","needn't","neednt","needs","neither","net","netscape","never","neverf","neverless","nevertheless","new","newer","newest","next","nf","ng","ni","nine","ninety","nl","no","no-one","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","notwithstanding","novel","now","nowhere","np","nr","nu","null","number","numbers","nz","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","om","omitted","on","once","one","one's","ones","only","onto","open","opened","opening","opens","opposite","or","ord","order","ordered","ordering","orders","org","other","others","otherwise","ought","oughtn't","oughtnt","our","ours","ourselves","out","outside","over","overall","owing","own","p","pa","page","pages","part","parted","particular","particularly","parting","parts","past","pe","per","perhaps","pf","pg","ph","pk","pl","place","placed","places","please","plus","pm","pmid","pn","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","pr","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provided","provides","pt","put","puts","pw","py","q","qa","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","reserved","respectively","resulted","resulting","results","right","ring","ro","room","rooms","round","ru","run","rw","s","sa","said","same","saw","say","saying","says","sb","sc","sd","se","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","seventy","several","sg","sh","shall","shan't","shant","she","she'd","she'll","she's","shed","shell","shes","should","should've","shouldn","shouldn't","shouldnt","show","showed","showing","shown","showns","shows","si","side","sides","significant","significantly","similar","similarly","since","sincere","site","six","sixty","sj","sk","sl","slightly","sm","small","smaller","smallest","sn","so","some","somebody","someday","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","sr","st","state","states","still","stop","strongly","su","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","sv","sy","system","sz","t","t's","take","taken","taking","tc","td","tell","ten","tends","test","text","tf","tg","th","than","thank","thanks","thanx","that","that'll","that's","that've","thatll","thats","thatve","the","their","theirs","them","themselves","then","thence","there","there'd","there'll","there're","there's","there've","thereafter","thereby","thered","therefore","therein","therell","thereof","therere","theres","thereto","thereupon","thereve","these","they","they'd","they'll","they're","they've","theyd","theyll","theyre","theyve","thick","thin","thing","things","think","thinks","third","thirty","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","till","tip","tis","tj","tk","tm","tn","to","today","together","too","took","top","toward","towards","tp","tr","tried","tries","trillion","truly","try","trying","ts","tt","turn","turned","turning","turns","tv","tw","twas","twelve","twenty","twice","two","tz","u","ua","ug","uk","um","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","upwards","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","uy","uz","v","va","value","various","vc","ve","versus","very","vg","vi","via","viz","vn","vol","vols","vs","vu","w","want","wanted","wanting","wants","was","wasn","wasn't","wasnt","way","ways","we","we'd","we'll","we're","we've","web","webpage","website","wed","welcome","well","wells","went","were","weren","weren't","werent","weve","wf","what","what'd","what'll","what's","what've","whatever","whatll","whats","whatve","when","when'd","when'll","when's","whence","whenever","where","where'd","where'll","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","whichever","while","whilst","whim","whither","who","who'd","who'll","who's","whod","whoever","whole","wholl","whom","whomever","whos","whose","why","why'd","why'll","why's","widely","width","will","willing","wish","with","within","without","won","won't","wonder","wont","words","work","worked","working","works","world","would","would've","wouldn","wouldn't","wouldnt","ws","www","x","y","ye","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","youll","young","younger","youngest","your","youre","yours","yourself","yourselves","youve","yt","yu","z","za","zero","zm","zr"]
stop_de = ["a","ab","aber","ach","acht","achte","achten","achter","achtes","ag","alle","allein","allem","allen","aller","allerdings","alles","allgemeinen","als","also","am","an","ander","andere","anderem","anderen","anderer","anderes","anderm","andern","anderr","anders","au","auch","auf","aus","ausser","ausserdem","außer","außerdem","b","bald","bei","beide","beiden","beim","beispiel","bekannt","bereits","besonders","besser","besten","bin","bis","bisher","bist","c","d","d.h","da","dabei","dadurch","dafür","dagegen","daher","dahin","dahinter","damals","damit","danach","daneben","dank","dann","daran","darauf","daraus","darf","darfst","darin","darum","darunter","darüber","das","dasein","daselbst","dass","dasselbe","davon","davor","dazu","dazwischen","daß","dein","deine","deinem","deinen","deiner","deines","dem","dementsprechend","demgegenüber","demgemäss","demgemäß","demselben","demzufolge","den","denen","denn","denselben","der","deren","derer","derjenige","derjenigen","dermassen","dermaßen","derselbe","derselben","des","deshalb","desselben","dessen","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","dir","doch","dort","drei","drin","dritte","dritten","dritter","drittes","du","durch","durchaus","durfte","durften","dürfen","dürft","e","eben","ebenso","ehrlich","ei","ei,","eigen","eigene","eigenen","eigener","eigenes","ein","einander","eine","einem","einen","einer","eines","einig","einige","einigem","einigen","einiger","einiges","einmal","eins","elf","en","ende","endlich","entweder","er","ernst","erst","erste","ersten","erster","erstes","es","etwa","etwas","euch","euer","eure","eurem","euren","eurer","eures","f","folgende","früher","fünf","fünfte","fünften","fünfter","fünftes","für","g","gab","ganz","ganze","ganzen","ganzer","ganzes","gar","gedurft","gegen","gegenüber","gehabt","gehen","geht","gekannt","gekonnt","gemacht","gemocht","gemusst","genug","gerade","gern","gesagt","geschweige","gewesen","gewollt","geworden","gibt","ging","gleich","gott","gross","grosse","grossen","grosser","grosses","groß","große","großen","großer","großes","gut","gute","guter","gutes","h","hab","habe","haben","habt","hast","hat","hatte","hatten","hattest","hattet","heisst","her","heute","hier","hin","hinter","hoch","hätte","hätten","i","ich","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","im","immer","in","indem","infolgedessen","ins","irgend","ist","j","ja","jahr","jahre","jahren","je","jede","jedem","jeden","jeder","jedermann","jedermanns","jedes","jedoch","jemand","jemandem","jemanden","jene","jenem","jenen","jener","jenes","jetzt","k","kam","kann","kannst","kaum","kein","keine","keinem","keinen","keiner","keines","kleine","kleinen","kleiner","kleines","kommen","kommt","konnte","konnten","kurz","können","könnt","könnte","l","lang","lange","leicht","leide","lieber","los","m","machen","macht","machte","mag","magst","mahn","mal","man","manche","manchem","manchen","mancher","manches","mann","mehr","mein","meine","meinem","meinen","meiner","meines","mensch","menschen","mich","mir","mit","mittel","mochte","mochten","morgen","muss","musst","musste","mussten","muß","mußt","möchte","mögen","möglich","mögt","müssen","müsst","müßt","n","na","nach","nachdem","nahm","natürlich","neben","nein","neue","neuen","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nie","niemand","niemandem","niemanden","noch","nun","nur","o","ob","oben","oder","offen","oft","ohne","ordnung","p","q","r","recht","rechte","rechten","rechter","rechtes","richtig","rund","s","sa","sache","sagt","sagte","sah","satt","schlecht","schluss","schon","sechs","sechste","sechsten","sechster","sechstes","sehr","sei","seid","seien","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","selbst","sich","sie","sieben","siebente","siebenten","siebenter","siebentes","sind","so","solang","solche","solchem","solchen","solcher","solches","soll","sollen","sollst","sollt","sollte","sollten","sondern","sonst","soweit","sowie","später","startseite","statt","steht","suche","t","tag","tage","tagen","tat","teil","tel","tritt","trotzdem","tun","u","uhr","um","und","uns","unse","unsem","unsen","unser","unsere","unserer","unses","unter","v","vergangenen","viel","viele","vielem","vielen","vielleicht","vier","vierte","vierten","vierter","viertes","vom","von","vor","w","wahr","wann","war","waren","warst","wart","warum","was","weg","wegen","weil","weit","weiter","weitere","weiteren","weiteres","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wer","werde","werden","werdet","weshalb","wessen","wie","wieder","wieso","will","willst","wir","wird","wirklich","wirst","wissen","wo","woher","wohin","wohl","wollen","wollt","wollte","wollten","worden","wurde","wurden","während","währenddem","währenddessen","wäre","würde","würden","x","y","z","z.b","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zu","zuerst","zugleich","zum","zunächst","zur","zurück","zusammen","zwanzig","zwar","zwei","zweite","zweiten","zweiter","zweites","zwischen","zwölf","über","überhaupt","übrigens"]


def flemmatize(flat_corpus, lang='en'):
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
                          'puerorum', 'besieger', 'mayë'])
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
        #flem = ld.flemmatize(entry)
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

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, workers=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = LdaMulticore(workers=workers,corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        perp = model.log_perplexity(corpus)
        coherence = coherencemodel.get_coherence()
        coherence_values.append(coherence)
        print('{} processed with coherence value {} and perplexity {}'.format(num_topics, coherence, perp))

    return model_list, coherence_values

def LDA(ds, limit=None, minTopics=2, maxTopics=50, step=6, workers=8, lang='en'):
    
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
    fig.savefig(save_path + '/LDA-coherence.png', dpi=100)
    
    max_coherence = max(coherence_values)
    best_result_index = coherence_values.index(max_coherence)
    optimal_model = model_list[best_result_index]
    num_topics = optimal_model.get_topics().shape[0]
    print('Optimal model has %d topics with coherence score %s and perplexity %s'
          % (num_topics, max_coherence, optimal_model.log_perplexity(corpus)))
    
    return optimal_model, corpus, id2word

def visualizeTopics(m, c, ids, mds='mmds'):
    # Visualize the topics
    vis = pyLDAvis.gensim_models.prepare(m, c, ids, mds=mds)
    return vis

def distributionalDiff(lda_model, corpus, ids, samples, lang='english'):
    samples = flemmatize(flatten_list(flatten_list(samples.map(join)['text'])), lang=lang)
    corpus2 = [ids.doc2bow(text) for text in samples]
    
    # Get the mean of all topic distributions in one corpus
    quatrain_topic_vectors = []
    for quatrain in corpus:
        quatrain_topic_vectors.append(lda_model.get_document_topics(quatrain, minimum_probability=0))
    quatrain_average = np.average(np.array(quatrain_topic_vectors), axis=0)

    samples_topic_vectors = []
    for quatrain in corpus2:
        samples_topic_vectors.append(lda_model.get_document_topics(quatrain, minimum_probability=0))
    samples_average = np.average(np.array(samples_topic_vectors), axis=0)

    # Calculate the distance between the distribution of topics in both corpora
    difference_of_distributions = np.linalg.norm(quatrain_average - samples_average)
    cosine = np.dot(quatrain_average[1], samples_average[1])/(norm(quatrain_average[1])*norm(samples_average[1]))
    jsd = distance.jensenshannon(quatrain_average[1], samples_average[1])
    
    return difference_of_distributions, cosine, jsd


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min_topics', type=int, default=50)
    parser.add_argument('--max_topics', type=int, default=250)
    parser.add_argument('--stepsize', type=int, default=10)
    args = parser.parse_args()



if args.lang == 'en':
    QuaTrain = load_from_disk(current_path + '/data/training_data/QuaTrain')
else:
    QuaTrain = load_from_disk(current_path + '/data/training_data/QuaTrain-de')

save_path = current_path + '/data/topic_modeling/trained_model/' + args.lang

if not os.path.exists(save_path):
    os.makedirs(save_path)

m, c, ids = LDA(QuaTrain, minTopics=args.min_topics, maxTopics=args.max_topics, limit=args.limit, 
                step=args.stepsize, lang=args.lang, workers=args.workers)

m.save(save_path + '/model')
corpora.MmCorpus.serialize(save_path + '/corpus.mm', c)

vis = visualizeTopics(m, c, ids, mds='mmds')
pyLDAvis.save_json(vis, save_path + '/vis/vis.json')
pyLDAvis.save_html(vis, save_path + '/vis/vis.html')



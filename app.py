import heapq
import itertools
import nltk
import re

from flask import Flask, render_template, request
#from nltk import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import pdb

app = Flask(__name__)

# The below is just used in development to force reloading of static assets
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Flask-WTF requires an encryption key.
# For testing we can use whatever, but before deploying 
# we'd need a real key in an env variable
# TO-DO: move this to an env variable
app.config['SECRET_KEY'] = 'abc123'

# If you want to add additional stopwords or other things to remove
# to the set provided by NLTK, add them here.
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ['also', 'across', 'within', '—', 'ad', 'hoc']

# how deep we'll go into the list of the most common words
COMMON_WORDS_CAP = 25


def process_text(text):
    """
    Helper function to reduce a ton of duplication down below, 
    since we basically do all of this twice, once for their text
    and then again for our own text.
    Takes the raw text from form and returns:
    - clean_text: text with punctuation and non-alpha characters stripped
    - sentences': tokenized sentences from text
    - sentence_scores: the sentences, scored by word weight,
    - words: the tokenized words from the text
    - most_common_words: a FreqDist counter/ranking of the words by most used
    - word_weights: the words given a weight score based on frequency
    - summary: A simplistic summary of the text based on sentence weights.
    """
    # First, do some quick pre-processing to clean newlines
    # and stray chars, then cast to lowercase.
    sentences = nltk.sent_tokenize(text.replace('\n', '.').replace('. .', '.'))
    clean_text = re.sub('[^a-zA-Z]', ' ', text.replace('\n', '').lower())
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    words = [word for word in word_tokenize(clean_text) if not word in stop_words]
    word_frequency = nltk.FreqDist(words).most_common()
    # Now extract the top of the Counter to get the number of times the most
    # common word appeared, then we can make most_common_words a dict
    max_frequency = word_frequency[0][1]
    word_frequency = dict(word_frequency)
    word_weights = {key: value/max_frequency for key, value in word_frequency.items()}
    
    # now we'll calc the relative weight of sentences
    sentence_scores = {}
    
    for s in sentences:
        s_words = [word for word in word_tokenize(s.lower()) if not word in stop_words]

        # Limit to shorter-ish sentences to avoid overly long keys and 
        # keyword stuffing. This is an arbitrary limit
        if len(s_words) < 30:
            for word in s_words:
                if word in word_frequency.keys():
                    if s not in sentence_scores.keys():
                        sentence_scores[s] = word_weights[word]
                    else:
                        sentence_scores[s] += word_weights[word]
    # and now build a summary
    summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    most_common_words = {k: v for k, v in word_frequency.items() if v > 1}
    return {
        'sentences': sentences,
        'sentence_scores': sentence_scores,
        'summary': summary,
        'words': words,
        'word_frequency': word_frequency,
        'word_weights': word_weights,
        'most_common_words': most_common_words
    }

def get_shared_words(their_common_words, your_common_words):
    """
    Finds commonalities and misses between their common words and our common words.
    It expects two dicts of their common words and ours, with the words and the usage count
    passed as key/value pairs.
    At this point, we only care about a word if the frequency > 1.
    Returns three items:
    - A dict of shared words and how much each of set uses the word
    - the overlap score, a percentage of how much the two sets overlap
    - misses, or words that don't match up from your text to theirs
    """
    common_dict = {}
    common_overlap = set(their_common_words.keys()) & set(your_common_words.keys())

    # now build out the common dict showing their usage count and ours, 
    # sort it and slice it to length
    for key in common_overlap:
        common_dict[key] = (their_common_words[key], your_common_words[key])
        common_dict = dict(sorted(common_dict.items(), key=lambda item: item[1], reverse=True))
        common_dict = dict(itertools.islice(common_dict.items(), COMMON_WORDS_CAP))

    # misses are words that are common in their text but not common in yours.
    # they may be present in yours, but they are not common.
    # Note that the floor here is words present at least 3 times to reduce noise.
    misses = {k: v for k, v in their_common_words.items() if k not in your_common_words.keys() and v > 2}
    misses = dict(itertools.islice(misses.items(), COMMON_WORDS_CAP))
    return {
        'common_overlap_score': float(len(common_overlap)) / len(their_common_words.keys()) * 100,
        'common_words_dict': common_dict,
        'misses': misses
    }

def get_phrases(words):
    """
    Extracts most commonly used phrases and returns them. 
    See https://stackoverflow.com/questions/2452982/how-to-extract-common-significant-phrases-from-a-series-of-text-entries
    
    We're going to get both two-word and three-word phrases.
    """
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    tri_finder = TrigramCollocationFinder.from_words(words)
    # only trigrams that appear 2+ times
    tri_finder.apply_freq_filter(2)

    bi_finder = BigramCollocationFinder.from_words(words)
    # only bigrams that appear 3+ times
    bi_finder.apply_freq_filter(2)

    # return 20 n-grams with the highest PMI
    bi_phrases = bi_finder.nbest(bigram_measures.pmi, 20) 
    tri_phrases = tri_finder.nbest(trigram_measures.pmi, 20)
    
    phrases = tri_phrases + bi_phrases     
    # quick cleanup
    phrases = [" ".join(phrase) for phrase in phrases][0:15]
    return phrases


@app.route("/", methods=['GET', 'POST'])
def index():
    """
    Create form to compare two blocks of text.
    Potential future enhancements: 
    * Uploads
    * PDF extraction
    """
    from .forms import PearForm

    form = PearForm()
    errors = form.errors
    
    if request.method == 'POST':
        # Get form submission, 
        their_dict = process_text(form.text1.data)
        your_dict = process_text(form.text2.data)

        # now what's the total overlap between our words and theirs?
        their_set = set(their_dict['words'])
        overall_overlap = their_set & set(your_dict['words'])
        overall_overlap_score = float(len(overall_overlap)) / len(their_set) * 100

        # now we'll hand off to get_shared_words() to figure out the word commonalities and misses
        shared_words = get_shared_words(their_dict['most_common_words'], your_dict['most_common_words'])

        # And then extract phrases
        their_dict['phrases'] = get_phrases(their_dict['words'])
        your_dict['phrases'] = get_phrases(your_dict['words'])

        # just as a helper, let's get your most common words that ARE NOT 
        # in the original text, trimmed, so we don't have to do it in the template
        preferred_words = {k:v for k,v in your_dict['most_common_words'].items() if k not in their_dict['most_common_words'].keys() and v > 2 }    
        sliced_preferred_words = dict(itertools.islice(preferred_words.items(), 25))

        templateData = {
            'form': form,
            'errors': errors,
            'overall_overlap_score': overall_overlap_score,
            'their_dict': their_dict,
            'your_dict': your_dict,
            'common_overlap_score': shared_words['common_overlap_score'],
            'common_dict': shared_words['common_words_dict'],
            'misses': shared_words['misses'],
            'your_preferred_words': sliced_preferred_words,
        }
        return render_template('index.html', **templateData)
    # not a post, just do simple return
    templateData = {
      'form': form,
      'errors': errors
    }
    return render_template('index.html', **templateData)


@app.route('/update-nltk')
def update_nltk():
    """
    Download and update the NLTK modules
    """
    import ssl

    # Resolve SSL issues that can stop your nltk download
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    return render_template('updated.html')

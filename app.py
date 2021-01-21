from flask import Flask, render_template, request
import gensim
import nltk
from nltk import FreqDist
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

app = Flask(__name__)
# Used in development to force reloading of static assets
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Flask-WTF requires an encryption key 
# The string can be anything.
# For testing we can use whatever,
# but before deploying we'd need a real, secure key
# in an env variable
# TO-DO: move this to an env variable
app.config['SECRET_KEY'] = 'abc123'

# If you want to add additional stopwords 
# to the set provided by NLTK, do it here
ADDITIONAL_STOPWORDS = []

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
    overall_overlap_result = None
    common_overlap_result = None
    their_dict = None
    your_dict = None
    
    #if form.validate_on_submit():
    if request.method == 'POST':
        # Get form submission and do some quick initial cleanup:
        # Strip newlines, cast to lowercase for consistency,
        # and in case of and/or construction, split to extract both words.
        their_text = form.text1.data.replace('\n', '').lower().replace('/', " ")
        your_text = form.text2.data.replace('\n', '').lower().replace('/', " ")
        # TO-DO: consider splitting on hyphens and/or use a regex, too.

        # Now we'll tokenize the sentences
        their_sentence_tokens = sent_tokenize(their_text)
        your_sentence_tokens = sent_tokenize(your_text)

        # Build our collection of stopwords and other unwanted items.
        # Note: consider a regex to strip punctuation more cleanly
        punctuation = ['(', ')', ';', '-', '–', '&', ':', '[', ']', ',', '.', '#', '%', "'s", '“', '”', '’']
        stop_words = stopwords.words('english')
        stop_words += ADDITIONAL_STOPWORDS
        
        # Then get the individual word tokens. We could probably do this in a list comprehension,
        # but since we need to clean it up, a loop seems easier to read.
        their_keywords = your_keywords = []
        for sentence in their_sentence_tokens:
            # first word tokenize, tossing out stopwords and punctuation.
            words = [word for word in word_tokenize(sentence) if not word in stop_words and not word in punctuation]
            # Now we're going to use NLTK to find root words (lemmatization)
            # See: https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
            lemmatizer = WordNetLemmatizer()
            their_lem_words = [lemmatizer.lemmatize(word) for word in words]
            # and now a second pass to pick up any verbs remaining and append it
            their_keywords.append([lemmatizer.lemmatize(word, pos='v') for word in their_lem_words])



        #their_tokens = [[w for w in word_tokenize(sentence)] for sentence in their_sentence_tokens]
        #your_tokens = [[w for w in word_tokenize(sentence)] for sentence in their_sentence_tokens]

        # quick list comprehensions to strip out unwanted punctuation and stopwords
        #their_words = [word for word in their_tokens if not word in stop_words and not word in punctuation]
        #your_words = [word for word in your_tokens if not word in stop_words and not word in punctuation]

        
        their_gs_dict = gensim.corpora.Dictionary(their_keywords)
        print(their_gs_dict.token2id)

        # now create the bag of words
        # usage of gensim here is based on 
        # https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp
        their_corpus = [dictionary.doc2bow(their_keywords) for their_keywords in their_keywords]

        #your_gs_dict = gensim.corpora.Dictionary(your_keywords)

        #print(their_gs_dict.token2id)

        # And now get the frequency of keywords
        their_common_words = nltk.FreqDist(their_keywords).most_common(25)
        your_common_words = nltk.FreqDist(your_keywords).most_common(25)

        # now what's the total overlap? Use the full word stack
        their_set = set(their_keywords)
        your_set = set(your_keywords)

        overall_overlap = their_set & your_set
        overall_overlap_result = float(len(overall_overlap)) / len(their_set) * 100

        # What about among the keywords?
        their_set = set([item[0] for item in their_common_words])
        your_set = set([item[0] for item in your_common_words])

        common_overlap = their_set & your_set
        common_overlap_result = float(len(common_overlap)) / len(their_set) * 100

        # OK, but keywords are nice,
        # but if we really want to be speaking their language,
        # we have to think about phrases.
        # See https://stackoverflow.com/questions/2452982/how-to-extract-common-significant-phrases-from-a-series-of-text-entries
        # We're going to get both two-word and three-word phrases
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()

        tri_finder = TrigramCollocationFinder.from_words(their_words)
        # only trigrams that appear 3+ times
        tri_finder.apply_freq_filter(3)

        bi_finder = BigramCollocationFinder.from_words(their_words)
        # only bigrams that appear 3+ times
        bi_finder.apply_freq_filter(3)

        # return the 10 n-grams with the highest PMI
        bi_phrases = bi_finder.nbest(bigram_measures.pmi, 10) 
        tri_phrases = tri_finder.nbest(bigram_measures.pmi, 10)
        their_phrases = tri_phrases + bi_phrases 
        # quick cleanup
        their_phrases = [" ".join(phrase) for phrase in their_phrases][0:10]

        # now do it again for your words
        tri_finder = TrigramCollocationFinder.from_words(your_words)
        # only trigrams that appear 3+ times
        tri_finder.apply_freq_filter(3)

        bi_finder = BigramCollocationFinder.from_words(your_words)
        # only bigrams that appear 3+ times
        bi_finder.apply_freq_filter(3)

        # return the 10 n-grams with the highest PMI
        bi_phrases = bi_finder.nbest(bigram_measures.pmi, 10) 
        tri_phrases = tri_finder.nbest(bigram_measures.pmi, 10)
        your_phrases = tri_phrases + bi_phrases 
        # quick cleanup
        your_phrases = [" ".join(phrase) for phrase in your_phrases][0:10]

        their_dict = {
            'text': their_text,
            'keywords': their_keywords,
            'most_common': their_common_words,
            'phrases': their_phrases
        }
        your_dict = {
            'text': your_text,
            'keywords': your_keywords,
            'most_common': your_common_words,
            'phrases': your_phrases
        }
    templateData = {
      'form': form,
      'errors': errors,
      'overlap': overall_overlap_result,
      'common_overlap': common_overlap_result,
      'their': their_dict,
      'your': your_dict
    }
    return render_template('index.html', **templateData)


@app.route('/update-nltk')
def update_nltk():
    """
    Download and update the NLTK modules
    """
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    return render_template('updated.html')

# ---------------------------------
# Dashboard for Yelp gross-o-meter
# ---------------------------------
import streamlit as st
st.title('Yelp gross-o-meter')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, math
import pickle as pkl
agg_model_directory = r'data/models/aggregate_model/'
import yelp_query

# NLP
import gensim.downloader
from gensim import matutils
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS = STOP_WORDS.union({'ll', 've','s','id'})

# language detection
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect, detect_langs


###############################################
# functions for model and feature engineering #
###############################################

@st.experimental_singleton
def load_models():
    model_load_state = st.text('Loading models...')
    # import GloVe word vectors (twitter dataset, 200 dimensions)
    gv = gensim.downloader.load('glove-twitter-200')
    model_load_state.text('Loading models... done!')
    return gv

gv = load_models()

# DON'T stem/lemmatize here because the glove_vectors vocabulary doesn't do that
def text_cleaner(s):
    s = re.sub('<[^<]+?>', '', s.lower()) # remove tags
    # remove stop words
    s = re.sub('[^a-z\s\n]', '', s)
    s = s.split()
    # spacy stop words are a little more complete than gsp stop words
    s = [w for w in s if not w in STOP_WORDS]
    return s

def apply_gv(doc, method='list'):
    """
        Grabs glove of each word of the review
        Returns a list of vectors
    """
    vecs = []
    keys = []
    for w in doc:
        if w in ('food','like'): # words that hurt us!
            continue
        try:
            vec = gv[w]
        except KeyError: # if glove doesn't have a vector
            continue
        vecs.append(vec)
        keys.append(w)
    if vecs: 
        vecs = np.asarray(vecs)
        return vecs
    if not vecs:
        return None
    
def vec_similarity(vec1, vec2):
    """
        Dot product of two unit vectors to get similarity between them, same implementation as gensim similarity
    """
    return np.dot(matutils.unitvec(vec1),matutils.unitvec(vec2))

def review2vec(text):
    """
        Cleans text, vectorizes words, measures cosine distance to selected DIRTY_WORDS and PEST_WORDS, and calculates aggregate features.
        Output is a dataframe of one row
    """
    vecs = apply_gv(text_cleaner(text), 'list')
    
    # corner case: all words are not in the vocabulary (i.e., different language, short reviews, typo-ridden reviews, etc.)
    if vecs is None:
        return pd.DataFrame([np.nan]) # still return something so we keep the indexing

    # dataframe for each review
    df = pd.DataFrame()
    df['vec'] = list(vecs)
    df['dirty_sim'] = df.vec.apply(lambda v: vec_similarity(DIRTY_VECTOR, v))
    df['pest_sim'] = df.vec.apply(lambda v: vec_similarity(PEST_VECTOR, v))
    df['both_sim'] = df[['dirty_sim','pest_sim']].max(axis=1)

    # average all word vectors to make a review vector
    df_agg = pd.DataFrame([np.mean(df.vec)])
    df_agg['both_sim'] = np.mean(df.both_sim)
    df_agg['pest_sim'] = np.mean(df.pest_sim)
    df_agg['dirty_sim'] = np.mean(df.dirty_sim)
    df_agg.columns = df_agg.columns.astype(str)
    return df_agg

def filter_reviews(review_list):
    """
        Removes reviews if they are < 10 words long or not in English
    """
    rl = pd.DataFrame({'text': review_list})
    # rl = rl[rl.text.apply(len) >= 10]
    rl['language'] = rl['text'].apply(detect)
    return rl[rl.language == 'en'].text.values

def review_list_2vec(review_list):
    """
        Applies review2vec to a list of reviews
    """
    review_list = filter_reviews(review_list)
    return pd.concat(map(review2vec, review_list)).reset_index(drop=True)

def agg_features(review_features):
    """
        Takes as input the output of review_list_2vec()
    """
    dff = pd.DataFrame(review_features.mean(axis=0)).T
    # dff['num_reviews'] = len(review_features)
    return dff

# these are a few representative words
DIRTY_WORDS = ('disgusting','smelly','rotten','nasty','gross','dirty','undercooked','moldy','puke','sick','unhealthy')
PEST_WORDS = ('rat','bugs','cockroach','fly','ant','flea','insect','infestation','infest')
# average vector of these words
DIRTY_VECTOR = np.mean(np.array([gv[w] for w in DIRTY_WORDS]),axis=0)
PEST_VECTOR = np.mean(np.array([gv[w] for w in PEST_WORDS]),axis=0)

model = pkl.load(open(agg_model_directory + 'mean_vectors_agg_model.pkl', 'rb'))
all_predicted_scores = np.loadtxt('data/processed data/predicted_business_insp_scores.csv', delimiter=',')

def percentile(score):
    return sum(all_predicted_scores < score)/len(all_predicted_scores)*100

def histogram_plot(score):
    """
        Plots the score in relation to the entire dataset
    """
    fig = plt.figure()#figsize=(7,4),dpi=100)
    bin_y, bin_x, bars = plt.hist(all_predicted_scores,bins=75,alpha=1,color='xkcd:sky blue')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.tick_params('y',left=False)
    bin_height = min(bin_y[np.argmin(np.abs(bin_x-score))],80)
    if score < min(bin_x):
        bin_height = 0
    plt.arrow(score, bin_height+13, 0, -10, length_includes_head=True,
              head_width=0.5, head_length=3, color='black')
    plt.xlabel('Score (/100)')
    plt.ylabel('# restaurants')
    plt.title('')
    return fig

    

################################################
#               dashboard things               #
################################################

yelp_name = st.text_input("Input a restaurant name here (try 'Super Bowl'):")
yelp_address = st.text_input("Input a restaurant address here (try '713 Cannon Dr'):")
if yelp_name and yelp_address:
    business_id, reviews = yelp_query.get_reviews(
        name = yelp_name,
        address = yelp_address
    )
    # feature engineering
    review_features = review_list_2vec(reviews)
    # aggregate features
    features = agg_features(review_features)
    # random forest model
    score = model.predict(features)[0]
    
    # best_review = reviews[np.argmin(features.both_sim)]
    best_review = reviews[np.argmax(model.predict(review_features))]
    text_display = 800
    if len(best_review)>text_display:
        best_review = best_review[:text_display] + " ..."
    
    # worst_review = reviews[np.argmax(features.both_sim)]
    worst_review = reviews[np.argmin(model.predict(review_features))]
    if len(worst_review)>text_display:
        worst_review = worst_review[:text_display] + " ..."
    
    st.header(f"[Yelp page](https://www.yelp.com/biz/{business_id})")
    st.subheader("Predicted inspection score: **{:.2f}/100**".format(score))
    pctle = percentile(score)
    st.write(yelp_name + ' is cleaner than **{:.2f}%** of all restaurants!'.format(pctle))
    fig = histogram_plot(score)
    st.pyplot(fig)
    st.write("**Best review:**")
    st.write(best_review)
    st.write("**Worst review:**")
    st.write(worst_review)


input_review = st.text_input('Input raw review text here:') 
if input_review:
    # feature engineering
    review_features = review_list_2vec([input_review])
    # aggregate features
    features = agg_features(review_features)
    # random forest model
    score = model.predict(features)[0]
    st.subheader("Predicted inspection score: **{:.2f}/100**".format(score))
    pctle = percentile(score)
    st.write('That\'s cleaner than **{:.2f}%** of all restaurants!'.format(pctle))

# ---------------------------------
# Dashboard for Yelp gross-o-meter
# ---------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, scipy, math, sklearn
import pickle as pkl
from gensim import matutils
import gensim.downloader
from spacy.lang.en.stop_words import STOP_WORDS
import yelp_query

data_directory = '/data'
processed_data_directory = r'data/processed data/'
model_directory = r'data/models/'

@st.experimental_singleton
def load_glove():
    model_load_state = st.text('Loading GloVe...')
    gv = gensim.downloader.load('glove-twitter-200')
    model_load_state.text('Loading GloVe... done!')
    return gv


st.title('Yelp gross-o-meter')

gv = load_glove()

# these are a few representative words
DIRTY_WORDS = ('disgusting','smelly','rotten','nasty','gross','dirty','undercooked','moldy','puke','sick','unhealthy')
PEST_WORDS = ('rat','bugs','cockroach','fly','ant','flea','insect','infestation','infest')
CLEAN_WORDS = ('clean','pristine','immaculate','organized','fresh')
# average vector of these words
DIRTY_VECTOR = np.mean(np.array([gv[w] for w in DIRTY_WORDS]),axis=0)
PEST_VECTOR = np.mean(np.array([gv[w] for w in PEST_WORDS]),axis=0)

## clean up text
STOP_WORDS = STOP_WORDS.union({'ll', 've','s','id'})
# DON'T stem/lemmatize here because the glove_vectors vocabulary doesn't do that
def text_cleaner(s):
    s = re.sub('<[^<]+?>', '', s.lower()) # remove tags
    # remove stop words
    s = re.sub('[^a-z\s\n]', '', s)
    s = s.split()
    # spacy stop words are a little more complete than gsp stop words
    s = [w for w in s if not w in STOP_WORDS]
    return s

def review2vec(doc):
    """
        Applies word2vec on each word of the review
        Returns a list of vectors, oops
    """
    vecs = []
    keys = []
    for w in doc:
        if w in ('food','like'):
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

def review_model_vec(text, method='top_percent_df', percent=0.15):
    """
        Cleans text, vectorizes words, measures cosine distance to selected DIRTY_WORDS and PEST_WORDS, and calculates aggregate features.
        Output is a dataframe of one row
    """
    vecs = review2vec(text_cleaner(text))
    
    # corner case: all words are not in the vocabulary (i.e., different language, short reviews, typo-ridden reviews, etc.)
    if vecs is None:
        return pd.DataFrame([np.nan]) # still return something so we keep the indexing

    # dataframe for each review
    df = pd.DataFrame()
    # df['words'] = words
    df['vec'] = list(vecs)
    df['dirty_sim'] = df.vec.apply(lambda v: vec_similarity(DIRTY_VECTOR, v))
    df['pest_sim'] = df.vec.apply(lambda v: vec_similarity(PEST_VECTOR, v))
    df['both_sim'] = df[['dirty_sim','pest_sim']].max(axis=1)

    # aggregate word vectors to make a review vector
    # average the vectors from the top {percent}% of words closest to PEST_VECTOR or DIRTY_VECTOR
    ll = len(df)
    if method == 'top_percent_df':
        # don't take too many words if the review is insanely long
        bound = min(math.ceil(ll*percent),25)
        top_percent_sim = df.sort_values('both_sim',ascending=False)[0:bound]
        df_agg = pd.DataFrame([np.mean(top_percent_sim.vec)])
        df_agg['both_sim'] = np.mean(top_percent_sim.both_sim)
        df_agg['pest_sim'] = np.mean(top_percent_sim.pest_sim)
        df_agg['dirty_sim'] = np.mean(top_percent_sim.dirty_sim)
        return df_agg
    elif method == 'mean_df':
        df_agg = pd.DataFrame([np.mean(df.vec)])
        df_agg['both_sim'] = np.mean(df.both_sim)
        df_agg['pest_sim'] = np.mean(df.pest_sim)
        df_agg['dirty_sim'] = np.mean(df.dirty_sim)
        return df_agg
    if method == 'max_df':
        # only take the vector of ONE word per review
        max_row = df.sort_values('both_sim',ascending=False).iloc[0]
        return pd.DataFrame(max_row[['vec','dirty_sim','pest_sim','both_sim']]).T.reset_index()
    else: #method == number
        bound = min(math.ceil(ll*percent),25)
        top_percent_sim = df.sort_values('both_sim',ascending=False)[0:bound]
        return np.mean(top_percent_sim.both_sim)

    

# groups = df.sort_values('both_sim',ascending=False).groupby('business_id')
# df_list = []
# for business_id, group in groups:
#     dff = pd.DataFrame()
#     # average the dirtiest 15% of reviews (minimum 5) from each business
#     ll = len(group)
#     bound = max(5,math.ceil(ll*0.15))
#     dff[business_id] = group[df.columns[8:]][:bound].mean(axis=0)
#     dff=dff.T
#     dff['num_reviews'] = ll
#     dff['Score'] = group.Score.unique().mean()
#     df_list.append(dff)
    
yelp_name = st.text_input('Input a restaurant name here:')
yelp_address = st.text_input('Input a restaurant address here:')
if yelp_name and yelp_address:
    # reviews = yelp_query.get_reviews(name = "Super Bowl", address="719 W William Cannon Dr Ste 103")
    # feature engineering
    business_id, reviews = yelp_query.get_reviews(name = yelp_name, address=yelp_address)
    features = pd.concat(map(lambda x: review_model_vec(x,'top_percent_df',0.1), reviews)).reset_index(drop=True)
    best_review = reviews[np.argmin(features.both_sim)]
    text_display = 800
    if len(best_review)>text_display:
        best_review = best_review[:text_display] + " ..."
    worst_review = reviews[np.argmax(features.both_sim)]
    if len(worst_review)>text_display:
        worst_review = worst_review[:text_display] + " ..."
    
    st.header(f"[Yelp page](https://www.yelp.com/biz/{business_id})")
    st.write("**Best review**")
    st.write(best_review)
    st.write("**Worst review**")
    st.write(worst_review)


input_review = st.text_input('Input raw review text here:') 
if input_review:
    features = review_model_vec(input_review,'top_percent_df',0.1)
    st.write(features)


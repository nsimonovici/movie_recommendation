#coding:utf-8

# 1) Importations and options

import sys
import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model, model_from_json

# Set options
parser = argparse.ArgumentParser(description='Select genre for training')
parser.add_argument(dest='movie_id', action='store',
                    help='give the movie id to get recommendations')
args = parser.parse_args()

# 2) Functions

def dynamic_std_print(string_to_print):
    """
    Printing string on standard output and refreshing output line
    """
    sys.stdout.write('\r')
    sys.stdout.write(string_to_print)
    sys.stdout.flush()

def isgenre(row, genre):
    """
    To be used with an 'apply' method in dataframe.
    Returns 1 if the input genre is found within the genres or the keywords of the dataframe's row, 0 otheriwse
    """
    isgenre = False
    if genre in row.genres_str:
        isgenre = True
    elif genre in row.kw_str:
        isgenre = True
    if isgenre:
        out = 1
    else:
        out = 0
    return out

def generate_batch(pairs, shape_movies, shape_keywords, n_positive=50, negative_ratio=1.0, classification=False):
    """
    Generate batches of samples for training
    """
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Get pairs set
    pairs_set = set(pairs)

    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (movie_id, kw_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (movie_id, kw_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_movie = random.randrange(shape_movies)
            random_kw = random.randrange(shape_keywords)
            
            # Check to make sure this is not a positive example
            if (random_movie, random_kw) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_movie, random_kw, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'movie': batch[:, 0], 'kw': batch[:, 1]}, batch[:, 2]

def movies_embedding_model(movies_shape, kw_shape, embedding_size=50, classification=False):
    """
    Model to embed books and wikilinks using the functional API.
    Trained to discern if a movie is of a particular genre
    """
    
    # Both inputs are 1-dimensional
    movie = Input(name = 'movie', shape = [1])
    kw = Input(name = 'kw', shape = [1])
    
    # Embedding the movies (shape will be (None, 1, 50))
    movie_embedding = Embedding(name = 'movie_embedding',
                                 input_dim = movies_shape,
                                 output_dim = embedding_size)(movie)
    
    
    # Embedding the keywords (shape will be (None, 1, 50))
    kw_embedding = Embedding(name = 'kw_embedding',
                            input_dim = kw_shape,
                            output_dim = embedding_size)(kw)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, kw_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [movie, kw], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [movie, kw], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

def train_embedding(pairs, shape_movies, shape_kw, n_positive=1000, emb_size=50, verb=1):
    """
    Train neural network
    """
    # Instantiate model and show parameters
    model = movies_embedding_model(shape_movies, shape_kw, emb_size)
 
    # Generate batch
    gen = generate_batch(pairs, shape_movies, shape_kw, n_positive, negative_ratio=2, classification=True)

    # Train
    h = model.fit_generator(gen, epochs=10, 
                            steps_per_epoch=len(pairs) // n_positive,
                            verbose=verb)
    return h, model

# 3) Data loading

# Getting current path
path = os.getcwd()
data_file = '/data/input_df_embedding.pkl'
pairs_file = '/data/pairs_training.pkl'
# Verifying data presence
try :
    data_raw = pickle.load(open(path + data_file, 'rb'))
    print("Movie dataset loaded, loading positive sample for training")
except FileNotFoundError :
    print("Please check if the file %s is in the 'data' folder at the current location" % data_file)
try :
    pairs = pickle.load(open(path + pairs_file, 'rb'))
    print("Positive samples loaded")
except FileNotFoundError :
    print("Please check if the file %s is in the 'data' folder at the current location" % pairs_file)

# Get splitted keywords in a serie
s = data_raw.keywords.str.lower().str.split(",").values.tolist()
# Get unique keywords
keywords = [keywords_out for keywords_in in s for keywords_out in keywords_in]
# Remove stopwords
stop_words = set(stopwords.words('english'))
keywords_tokens_sw = [w for w in keywords if not w in stop_words]
set_keywords = set(keywords_tokens_sw)
unique_keywords = list(set_keywords)
# Building dictionnary
kw_index = {kw: idx for idx, kw in enumerate(unique_keywords)}
# Building reverse dictionnary
index_kw = {idx: kw for kw, idx in kw_index.items()}

# 4) Model implementation

print("Set neural network embedding model")
# Instantiate model
# Set n_positive
n_positive = 1000

# Load or run model
try:
    # load json and create model
    json_file = open(path + '/data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emb_model = model_from_json(loaded_model_json)
    # load weights into new model
    emb_model.load_weights(path + "/data/model.h5")
    print("Loaded model from disk")
except FileNotFoundError:
    history, emb_model = train_embedding(pairs, data_raw.shape[0], len(unique_keywords), n_positive, emb_size=100)
    # serialize model to JSON
    model_json = emb_model.to_json()
    with open(path + "/data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    emb_model.save_weights(path + "/data/model.h5")
    print("Saved model to disk")

# 5) Recommendation

# Get user input : movie id
try :
    movie_id = int(args.movie_id)
except ValueError :
    print("ERROR : please check that you entered an integer as film's id")
    exit(1)

print("Extracting genre's embedding layer weights")
# Extract genres embeddings
movie_layer = emb_model.get_layer('movie_embedding')
movie_weights = movie_layer.get_weights()[0]

# Normalize weights
movie_weights = movie_weights / np.linalg.norm(movie_weights, axis = 1).reshape((-1, 1))

# Calculate dot product between movie and all others
dists = np.dot(movie_weights, movie_weights[movie_id])

# Sort distance indexes from smallest to largest
sorted_dists = np.argsort(dists)

# Take the last n sorted distances
closest = sorted_dists[-11:]
closest = [x for x in reversed(closest)]
reco = data_raw.iloc[closest, :].title

# Display recommendations
print("You're selected movie :")
print(data_raw.title.loc[movie_id])
print("You're recommendations :")
for recom in reco.values[1:]:
    print(recom)

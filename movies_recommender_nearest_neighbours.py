#coding:utf-8

# 1) Importations and options

import os
import pandas as pd
import numpy as np
from sklearn import neighbors, preprocessing
import argparse
import pickle

# Set options
parser = argparse.ArgumentParser(description='Select genre for training')
parser.add_argument(dest='movie_id', action='store',
                    help='give the movie id to get recommendations')
args = parser.parse_args()

# 2) Functions

def load_db(path) :
    # Load main DB
    try:
        # file_name = 'https://s3.eu-west-2.amazonaws.com/filmsprojetrecommandation/reduced_data.csv'
        file_name = '/data/input_df_embedding.pkl'
        out = pickle.load(open(path + file_name, 'rb'))
    except FileNotFoundError: 
        print("ERROR : please check that '%s' is correct file" % file_name)
        exit(1)
    return out

def retrieve_one_hot_result(categ_ref, df):
    # Retrieve categories names
    categories = categ_ref.columns
    if isinstance(df, pd.core.frame.DataFrame):
        out_df = pd.DataFrame()
        # Separate DF in SERIES and apply SERIES function
        for row in df.iterrows() :   
            temp_ser = row[1]
            # Retrieve values of one hot encoding
            one_hot_ref = df[df == 1].dropna()
            # Replace values of categorial df with columns (one hot) names
            new_value = []
            for i in range(len(one_hot_ref.index)) :
                new_value.append(one_hot_ref.index[i].replace(categories[:len(one_hot_ref.index)][i] + "_",""))
            one_hot_ref.loc[:] = new_value
            # Replaces columns names of categorial df with original categories names
            one_hot_ref.index = categories[:len(one_hot_ref.index)]
            # Full serie
            out_df = out_df.append(one_hot_ref)

    elif isinstance(df, pd.core.series.Series) :
        # Retrieve values of one hot encoding
        one_hot_ref = df[df == 1].dropna()
        # Replace values of categorial df with columns (one hot) names
        new_value = []
        for i in range(len(one_hot_ref.index)) :
            new_value.append(one_hot_ref.index[i].replace(categories[:len(one_hot_ref.index)][i] + "_",""))
        one_hot_ref.loc[:] = new_value
        # Replaces columns names of categorial df with original categories names
        one_hot_ref.index = categories[:len(one_hot_ref.index)]
        # Full df
        out_df = one_hot_ref
        
    return out_df

metric="euclidean"
# Get user input : movie id
try :
    movie_id = int(args.movie_id)
except ValueError :
    print("ERROR : please check that you entered an integer as film's id")
    exit(1)

# Getting current path
path = os.getcwd()

# Load main DB
print("Loading dataset")
main_db = load_db(path)

# One Hot Encoding categorical variables
print("Encoding features")
# categorical_df = main_db.loc[:, ['movie_title', 'genre_1', 'genre_2','genre_3','genre_4','genre_5', 'plot_keywords']]
categorical_df = main_db.loc[:, ['title', 'genres', 'keywords']]
categorical_df_encoded = pd.get_dummies(categorical_df)

# Implement the algorithm, Look for the N nearest neighbors
neighbs = 6

# Euclidean distance nearest neighbors recommendation
print("Looking for nearest neighbours")
# nn_model_file = '/data/nearest_neigbhors_model.pkl'
# nn_algorithm = pickle.load(open(path + nn_model_file, 'rb'))
nn_algorithm = neighbors.NearestNeighbors(neighbs)
nn_algorithm.fit(categorical_df_encoded)

# Save fitted model for easy re-use
# pickle.dump(nn_algorithm, open(path + '/data/nearest_neigbhors_model.pkl', 'wb'))

# Look for the nearest neighbors of selected movie
movie_reference = pd.DataFrame([categorical_df_encoded.iloc[movie_id, :]])
    
resu = nn_algorithm.kneighbors(movie_reference)

# Concatenate outputs (keep only positive columns for categorical encoded columns)
print("Processing prediction")
out_df = pd.DataFrame()
for i in range(len(resu[1][0])) :
    out_df = out_df.append(retrieve_one_hot_result(categorical_df, categorical_df_encoded.iloc[resu[1][0][i], :]), ignore_index=False)
# Some cosmetic modifications
out_df = out_df.rename(index={out_df.index[0]:'REF'})
#    features = [x for x in out_df.columns if x.startswith('genre')]
#    features.append('movie_title')

string_out = "You provided with the movie : %s \n" % out_df.iloc[0,:].loc['title']
string_out = string_out + 'These are the movies recommendations based on the reference you provided (%s metric) :\n' % metric
count = 1
for row in out_df.iterrows() :
    if row[1].name == 'REF' :
        pass
    else :        
        string_out = string_out + str(count) + " : " + row[1].loc['title'] + "|\n"
        count += 1   

print(string_out)
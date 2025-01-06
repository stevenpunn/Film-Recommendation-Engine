import pandas as pd
import numpy as np
import re
import ipywidgets as widgets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display

# read from the spread sheet with data and create dataframe
movies = pd.read_csv("movies.csv")

# clean the titles to make easier search
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(clean_title)

# create a term frequency * inverse doc frequency matrix (TFIDF)
vectorizer = TfidfVectorizer(ngram_range =(1,2)) # looks at title as chunks of 2 words
tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

movie_input = widgets.Text(value="Enter Film", description="Movie Title", disabled=False)

movie_list = widgets.Output()
def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')
display(movie_input, movie_list)
ratings = pd.read_csv("ratings.csv")
movie_id = 0

similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 5)]["userId"].unique()
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
similar_user_recs = similar_user_recs[similar_user_recs > .1]
all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]

# find what % of all users recommend this movie
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

# compare percentages
rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis = 1)
rec_percentages.columns = ["similar", "all"]

rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
rec_percentages = rec_percentages.sort_values("score", ascending = False)
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

def find_similar_movies(movie_id):
    similar_user = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 5)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_user)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis = 1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending = False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

movie_input_name = widgets.Text(value = "Type Film Name", description = "Movie Title:", disabled = False)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_input_name.observe(on_type, names ="value")
display(movie_input_name, recommendation_list)

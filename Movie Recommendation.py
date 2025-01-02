# Use netflix dataset that contains info about the movies
# Take things such as the cast, title, description, and put into text
# Llama 2 will take the strings and embed them into vector space
# Sort into a vector storage using FAISS

import pandas as pd
df = pd.read_csv('netflix_titles.csv')

def create_textual_representation(row):
    textual_representation = f"""Type: {row['type']},
    Title: {row['title']},
    Diretor: {row['director']},
    Cast: {row['cast']},
    Release Year: {row['release_year']},
    Genres: {row['genre']},

    Description: {row['description']},
    """

    return textual_representation

df.apply(create_textual_representation, axis =1)
    
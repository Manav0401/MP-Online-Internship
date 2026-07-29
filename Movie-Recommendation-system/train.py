import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

ratings = ratings.sample(n=500000, random_state=42)

movies_with_ratings = ratings.merge(movies, on="movieId")

movie_stats = movies_with_ratings.groupby("title").agg(
    average_rating=("rating", "mean"),
    number_of_ratings=("rating", "count")
)

popular_movies = movie_stats[movie_stats["number_of_ratings"] >= 50].reset_index()

movies_df = movies_with_ratings.merge(popular_movies, on="title")

movie_matrix = movies_df.pivot_table(
    index="title",
    columns="userId",
    values="rating"
).fillna(0)

movie_matrix = movie_matrix.sort_index()

similarity = cosine_similarity(movie_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.index,
    columns=movie_matrix.index
)

os.makedirs("model", exist_ok=True)

with open("model/similarity.pkl", "wb") as f:
    pickle.dump(similarity_df, f)

with open("model/movies.pkl", "wb") as f:
    pickle.dump(popular_movies, f)

with open("model/movie_list.pkl", "wb") as f:
    pickle.dump(movie_matrix.index.tolist(), f)

print("Training Completed Successfully")
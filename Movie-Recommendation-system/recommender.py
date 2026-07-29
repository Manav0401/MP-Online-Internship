import pickle
import pandas as pd

with open("model/similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

with open("model/movies.pkl", "rb") as f:
    movies = pickle.load(f)

with open("model/movie_list.pkl", "rb") as f:
    movie_list = pickle.load(f)


def recommend(movie_name, n=10):

    if movie_name not in similarity_df.index:
        return pd.DataFrame(columns=[
            "title",
            "similarity",
            "average_rating",
            "number_of_ratings"
        ])

    recommendations = (
        similarity_df[movie_name]
        .sort_values(ascending=False)
        .iloc[1:n+1]
        .reset_index()
    )

    recommendations.columns = [
        "title",
        "similarity"
    ]

    recommendations = recommendations.merge(
        movies,
        on="title"
    )

    return recommendations
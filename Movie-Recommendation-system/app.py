from flask import Flask, render_template, request
from recommender import recommend, movie_list

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        movie_list=movie_list
    )


@app.route("/recommend", methods=["POST"])
def recommendation():

    movie = request.form["movie"]

    recommendations = recommend(movie)

    return render_template(
        "recommendations.html",
        movie=movie,
        recommendations=recommendations.to_dict("records")
    )


if __name__ == "__main__":
    app.run(debug=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import movie and ratings data
movies = pd.read_csv("HW3/Task1a/movies.csv")
ratings = pd.read_csv("HW3/Task1b/ratings.csv")

# split genres properly
movies_genres = movies.assign(
    genre=movies['genres'].str.split('|')
).explode('genre')

# remove junk genre
movies_genres = movies_genres[movies_genres['genre'] != '(no genres listed)']

# join ratings with movies to get genre information
movie_ratings = ratings.merge(movies_genres[['movieId', 'genre']], on='movieId', how='inner')

# fix: only have top genres displayed
top_genres = movie_ratings['genre'].value_counts().nlargest(10).index
movie_ratings = movie_ratings[movie_ratings['genre'].isin(top_genres)]

# order by median ratings
genre_order = movie_ratings.groupby('genre')['rating'].median().sort_values(ascending=False).index

# Plot: How do MovieLens reviews vary by genre?
plt.figure(figsize=(10,6))

sns.boxplot(
    data=movie_ratings,
    x='genre',
    y='rating',
    order=genre_order,
    palette=sns.color_palette("pastel"),
    fliersize=2
)

plt.title("MovieLens Reviews by Genre", fontsize=15)
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Rating", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

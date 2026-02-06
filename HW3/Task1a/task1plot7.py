import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

movies = pd.read_csv("HW3\\Task1a\\movies.csv")

# extract year
movies["year"] = movies["title"].transform(lambda x: x.strip()[-5:-1])

# filter out movies with no years
movies = movies[movies["year"].str.isdigit()]

# int
movies["year"] = movies["year"].astype(int)

# since some movies are outliers before 1915, bin them
movies["year_bin"] = movies["year"].where(movies["year"] >= 1915, "<1915")
year_order = ["<1915"] + sorted(movies.loc[movies["year"] >= 1915, "year"].unique())
movies["year_bin"] = pd.Categorical(movies["year_bin"], categories=year_order, ordered=True)

# singular genre tags
movies["genres"] = movies["genres"].str.split("|")
clean_movies = movies.explode("genres")

# create heatmap data
heatmap_data = pd.crosstab(clean_movies["year_bin"], clean_movies["genres"])

plt.figure(figsize=(12, 10))

# heatmap
ax = sb.heatmap(heatmap_data, cmap = "mako")

# specify y-ticks: every 5th row, including last row
n = len(heatmap_data)
ytick_idx = list(range(0, n, 5))
if (n - 1) not in ytick_idx:
    ytick_idx.append(n - 1)
ytick_idx = sorted(ytick_idx)
ax.set_yticks([i + 0.5 for i in ytick_idx])
ax.set_yticklabels(heatmap_data.index[ytick_idx])

plt.title("How do the number of movie releases and genres vary across years?", fontsize=15)
plt.ylabel("Year of Release", fontsize=12)
plt.xlabel("Genre", fontsize=12)
plt.xticks(rotation = 45, ha = "right")
plt.tight_layout()
plt.show()

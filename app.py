import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Netflix Movie Recommender", layout="wide")

# ---------------- LOAD DATA ----------------
movies = pickle.load(open('movies.pkl', 'rb'))

# ---------------- CREATE VECTORS ----------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ---------------- TRAIN MODEL ----------------
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(vectors)

# ---------------- TMDB API KEY ----------------
API_KEY = "YOUR_TMDB_API_KEY"

# ---------------- FETCH POSTER ----------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        data = requests.get(url).json()

        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
        else:
            return "https://via.placeholder.com/300x450.png?text=No+Poster"

    except:
        return "https://via.placeholder.com/300x450.png?text=No+Poster"

# ---------------- UI ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #141414;
        color: white;
    }
    h1 {
        color: red;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Netflix Movie Recommender")

selected_movie = st.selectbox("Select Movie", movies['title'].values)

# ---------------- BUTTON ----------------
if st.button("Recommend"):

    row = movies[movies['title'] == selected_movie].iloc[0]
    index = movies[movies['title'] == selected_movie].index[0]

    # ---------------- Similar Movies ----------------
    st.subheader("🔥 Similar Movies")

    distances, indices = model.kneighbors([vectors[index]])

    cols = st.columns(5)

    for count, i in enumerate(indices[0][1:6]):
        with cols[count]:
            st.image(fetch_poster(movies.iloc[i].movie_id))
            st.write(movies.iloc[i].title)

    # ---------------- Same Genre ----------------
    if len(row['genres']) > 0:
        genre = row['genres'][0]

        st.subheader("🎭 Same Genre")

        genre_movies = movies[movies['genres'].apply(lambda x: genre in x)].head(5)

        cols = st.columns(5)

        for count, (_, r) in enumerate(genre_movies.iterrows()):
            with cols[count]:
                st.image(fetch_poster(r.movie_id))
                st.write(r.title)

    # ---------------- Same Actor ----------------
    if len(row['cast']) > 0:
        actor = row['cast'][0]

        st.subheader("🌟 Same Actor")

        actor_movies = movies[movies['cast'].apply(lambda x: actor in x)].head(5)

        cols = st.columns(5)

        for count, (_, r) in enumerate(actor_movies.iterrows()):
            with cols[count]:
                st.image(fetch_poster(r.movie_id))
                st.write(r.title)

    # ---------------- Same Director ----------------
    if len(row['crew']) > 0:
        director = row['crew'][0]

        st.subheader("🎬 Same Director")

        director_movies = movies[movies['crew'].apply(lambda x: director in x)].head(5)

        cols = st.columns(5)

        for count, (_, r) in enumerate(director_movies.iterrows()):
            with cols[count]:
                st.image(fetch_poster(r.movie_id))
                st.write(r.title)

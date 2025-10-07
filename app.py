import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Smart Movie Recommender", page_icon="🎬")
st.title("🎬 Smart Movie Recommendation System")

# Load data
movies_dict = pickle.load(open('model/movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

def recommend(movie):
    try:
        movie_index = movies[movies['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return ["Movie not found. Try another."]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

movie_name = st.text_input("Enter a movie name:")
if st.button("Show Recommendations"):
    recs = recommend(movie_name)
    st.subheader("Top Recommendations:")
    for r in recs:
        st.write("🎥", r)

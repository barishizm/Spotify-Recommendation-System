import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="🎵 Spotify Recommender", page_icon="🎵", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #191414; color: white; }
    .stButton>button { background-color: #1DB954; color: white; border-radius: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Spotify Song Recommender")
st.markdown("Enter a song and discover similar tracks!")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.dropna()
    
    features = ['popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    scaled_array = df_scaled[features].values
    
    return df, df_scaled, scaled_array, features

df, df_scaled, scaled_array, features = load_data()

# --- Recommendation Function ---
def recommend_songs(song_name, artist_name, same_genre, n=10, pop_weight=0.3):
    matches = df[df['track_name'].str.lower() == song_name.lower()]
    if artist_name:
        matches = matches[matches['artists'].str.lower().str.contains(artist_name.lower())]
    
    if matches.empty:
        return None, None
    
    idx = matches.index[0]
    genre = df.loc[idx, 'track_genre']
    
    weights = np.array([0.5, 1.5, 1.5, 1.0, 0.5, 1.2, 0.8, 0.5, 1.5, 1.0])
    weighted_array = scaled_array * weights
    
    if same_genre:
        genre_mask = df['track_genre'] == genre
        genre_indices = df[genre_mask].index.tolist()
        song_vector = weighted_array[idx].reshape(1, -1)
        sim_scores = cosine_similarity(song_vector, weighted_array[genre_mask])[0]
        top_local = np.argsort(sim_scores)[::-1]
        top_local = [i for i in top_local if genre_indices[i] != idx]
        top_indices = [genre_indices[i] for i in top_local]
        sim_values = [sim_scores[i] for i in top_local]
    else:
        song_vector = weighted_array[idx].reshape(1, -1)
        sim_scores = cosine_similarity(song_vector, weighted_array)[0]
        top_indices = list(np.argsort(sim_scores)[::-1])
        top_indices = [i for i in top_indices if i != idx]
        sim_values = [sim_scores[i] for i in top_indices]
    
    seen = set()
    filtered_indices, filtered_scores = [], []
    for i, s in zip(top_indices, sim_values):
        name = df.loc[i, 'track_name'].lower()
        if name not in seen:
            seen.add(name)
            filtered_indices.append(i)
            filtered_scores.append(s)
        if len(filtered_indices) == n:
            break
    
    result = df.iloc[filtered_indices][['track_name', 'artists', 'track_genre', 'popularity']].copy()
    result['similarity_score'] = [round(s, 3) for s in filtered_scores]
    
    # Combined score
    sim_weight = 1 - pop_weight
    result['combined_score'] = (result['similarity_score'] * sim_weight) + (result['popularity'] * pop_weight)
    result = result.sort_values('combined_score', ascending=False)
    result['popularity_pct'] = (result['popularity'] / df['popularity'].max() * 100).astype(int).clip(0, 100)
    
    return result, genre

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    song_input = st.text_input("🎵 Song Name", placeholder="Shape of You")
with col2:
    artist_input = st.text_input("🎤 Artist (optional)", placeholder="Ed Sheeran")

same_genre = st.toggle("Only recommend same genre", value=True)

popularity_weight = st.slider("⭐ Popularity vs Similarity", 0, 100, 30,
                               help="Higher = more popular songs first")
similarity_weight = 100 - popularity_weight
st.caption(f"Similarity: {similarity_weight}%  —  Popularity: {popularity_weight}%")

n_recs = st.slider("Number of recommendations", 5, 20, 10)

if st.button("🔍 Get Recommendations"):
    if song_input:
        with st.spinner("Finding similar songs..."):
            results, genre = recommend_songs(song_input, artist_input, same_genre, n_recs, popularity_weight / 100)
        
        if results is None:
            st.error(f"'{song_input}' not found. Please check the song name.")
        else:
            st.success(f"Recommendations for **{song_input}** — Genre: `{genre}`")
            
            for _, row in results.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([3, 2, 1])
                    with c1:
                        st.markdown(f"**{row['track_name']}**")
                        st.caption(row['artists'])
                    with c2:
                        st.caption(f"🎸 {row['track_genre']}")
                        st.progress(int(min(max(row['popularity_pct'], 0), 100)))
                    with c3:
                        st.metric("Similarity", f"{row['similarity_score']}")
                    st.divider()
    else:
        st.warning("Please enter a song name!")
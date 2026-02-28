# Spotify Song Recommendation System

A content-based music recommendation system built with Python and Streamlit. Enter any song and discover similar tracks based on audio features.

🔗 **Live Demo:** [spotify-recommendation-sys-1.streamlit.app](https://spotify-recommendation-sys-1.streamlit.app)

---

## 📸 Preview

> Enter a song name, adjust your preferences, and get instant recommendations!

---

## 🚀 Features

- 🎯 **Content-Based Filtering** — recommends songs based on audio similarity
- ⭐ **Popularity Weighting** — balance between similarity and popularity with a slider
- 🎸 **Genre Filtering** — optionally restrict recommendations to the same genre
- 🔁 **Duplicate Removal** — no repeated songs in results
- ⚡ **Fast & Lightweight** — no full similarity matrix, computes on demand

---

## 🧠 How It Works

1. **Data Cleaning** — removed missing values from the Spotify dataset
2. **Feature Scaling** — normalized all audio features to [0, 1] using MinMaxScaler
3. **Feature Weighting** — key musical features (energy, valence, danceability) are weighted higher
4. **Cosine Similarity** — computed between the input song and all other tracks
5. **Combined Score** — final ranking blends similarity score and popularity

```
combined_score = (similarity × sim_weight) + (popularity × pop_weight)
```

---

## 🎛️ Audio Features Used

| Feature | Description |
|---|---|
| `danceability` | How suitable a track is for dancing |
| `energy` | Perceptual measure of intensity |
| `valence` | Musical positiveness of a track |
| `acousticness` | Whether the track is acoustic |
| `instrumentalness` | Predicts if a track has no vocals |
| `speechiness` | Presence of spoken words |
| `liveness` | Presence of a live audience |
| `loudness` | Overall loudness in decibels |
| `tempo` | Estimated tempo in BPM |
| `popularity` | Spotify popularity score (0–100) |

---

## 📦 Installation

```bash
git clone https://github.com/barishizm/Spotify-Recommendation-System.git
cd Spotify-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
Spotify-Recommendation-System/
├── app.py                              # Streamlit application
├── dataset.csv                         # Cleaned Spotify dataset
├── Spotify_Recommend_System_Preprocess.ipynb  # Data preprocessing notebook
├── requirements.txt                    # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data processing
- **Scikit-learn** — feature scaling & cosine similarity
- **Streamlit** — web interface
- **Streamlit Cloud** — deployment

---

## 📊 Dataset

- 114,000+ songs across 114 genres
- Source: [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Each genre contains exactly 1,000 tracks for a balanced distribution

---

## 👤 Author

**barishizm** — [github.com/barishizm](https://github.com/barishizm)

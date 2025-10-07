import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load dataset
movies = pd.read_csv('data/movies.csv')

# Create a 'tags' column combining text data
movies['tags'] = movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['overview']

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Save model artifacts
pickle.dump(movies.to_dict(), open('model/movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
print("✅ Model built and saved successfully!")

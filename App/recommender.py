import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import joblib
import os

class MusicRecommender:
    def __init__(self):
        self.similarity_scores = {}
        self.kmeans_labels = None
        self.X_pca = None
        self.X_tsne = None
        self.data = None
        self.raw_data=None
        self.features=['acousticness', 'danceability', 'energy',
                              'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'duration_ms','release_year']

    def fit(self, data_path='top_tracks_data_new.csv'):
        # Step 1: Data Preprocessing
        self.raw_data = pd.read_csv(data_path).drop_duplicates(subset=['id'])
        self.data = pd.read_csv(data_path).drop_duplicates(subset=['id'])
        if 'Unnamed: 0' in self.data.columns:
            self.data.drop(columns=['Unnamed: 0'], inplace=True)
        # Handle missing values if any
        self.data.dropna(inplace=True)

        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = ['acousticness', 'danceability', 'energy',
                              'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'duration_ms','release_year']
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        # Step 2: Feature Selection (We'll use all numerical features)
        selected_features = numerical_features

        # Step 3: Similarity Calculation
        similarity_functions = {
            'euclidean_distances': euclidean_distances,
            'manhattan_distances': manhattan_distances,
            'cosine_distances': cosine_distances,
            'pairwise_distances': pairwise_distances
        }

        for similarity_name, similarity_func in similarity_functions.items():
            if similarity_name == 'pairwise_distances':
                self.similarity_scores[similarity_name] = similarity_func(self.data[selected_features])
            else:
                self.similarity_scores[similarity_name] = similarity_func(self.data[selected_features])

        # Fit KMeans
        self.kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(self.data[selected_features])

        # Apply PCA and t-SNE
        pca = PCA(n_components=7, random_state=42)
        self.X_pca = pca.fit_transform(self.data[selected_features])
        tsne = TSNE(n_components=3, random_state=42)
        self.X_tsne = tsne.fit_transform(self.data[selected_features])

    def save_model(self, filepath):
        if os.path.exists(filepath):
            print(f"Model file {filepath} already exists. Skipping save operation.")
            return
        
        print(f"Saving model to {filepath}")
        
        # Save the model using joblib
        joblib.dump({
            'similarity_scores': self.similarity_scores,
            'kmeans_labels': self.kmeans_labels,
            'X_pca': self.X_pca,
            'X_tsne': self.X_tsne,
            'features': self.features
        }, filepath)

    @classmethod
    def load_model(cls, filepath, data_path='top_tracks_data_new.csv'):
        print(f"Loading model from {filepath}...")
        
        try:
            # Load the model using joblib
            model_dict = joblib.load(filepath)
            
            print(f"Keys in loaded model dictionary: {model_dict.keys()}")
            
            if model_dict is None:
                print("Error: Loaded model dictionary is None.")
                return None
            
            print("Model dictionary loaded successfully.")
            
            # Create a new instance of MusicRecommender
            recommender = cls()
            
            # Load and set the data
            recommender.raw_data = pd.read_csv(data_path).drop_duplicates(subset=['id'])
            recommender.data = pd.read_csv(data_path).drop_duplicates(subset=['id'])
            if 'Unnamed: 0' in recommender.data.columns:
                recommender.data.drop(columns=['Unnamed: 0'], inplace=True)
            recommender.data.dropna(inplace=True)
            
            # Populate the instance with the saved attributes
            recommender.similarity_scores = model_dict.get('similarity_scores')
            recommender.kmeans_labels = model_dict.get('kmeans_labels')
            recommender.X_pca = model_dict.get('X_pca')
            recommender.X_tsne = model_dict.get('X_tsne')
            recommender.features = model_dict.get('features')
            
            print("Model attributes and data populated successfully.")
            
            return recommender
        
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return None


    def get_song_index_by_name(self, song_name):
        if song_name not in self.data['track_name'].unique():
            print(f"Error: {song_name} is not a valid song name.")
            return None
        i = self.data[self.data['track_name'] == song_name].index[0]
        return i

    def recommend(self, song_name, n=10):
        song_index = self.get_song_index_by_name(song_name)
        recommendations = {}
        
        print(f"Checking attributes before recommendations:")
        print(f"self.similarity_scores: {self.similarity_scores}")
        print(f"self.kmeans_labels: {self.kmeans_labels}")
        print(f"self.X_pca: {self.X_pca}")
        print(f"self.X_tsne: {self.X_tsne}")
        
        if self.similarity_scores is None:
            print("Error: similarity_scores is not populated.")
            return
        
        if self.kmeans_labels is None:
            print("Error: kmeans_labels is not populated.")
            return
        
        if self.X_pca is None:
            print("Error: X_pca is not populated.")
            return
        
        if self.X_tsne is None:
            print("Error: X_tsne is not populated.")
            return
        
        for similarity_name, similarity_scores in self.similarity_scores.items():
            if similarity_scores is None:
                print(f"Error: similarity_scores[{similarity_name}] is None.")
                continue
                
            scores = similarity_scores[song_index]
            if scores is None:
                print(f"Error: scores for {similarity_name} is None.")
                continue
                
            similar_song_indices = scores.argsort()
            similar_song_indices = similar_song_indices[similar_song_indices != song_index]
            recommendations[similarity_name] = similar_song_indices[:n]

        # KMeans recommendation
        cluster_label = self.kmeans_labels[song_index]
        cluster_indices = np.where(self.kmeans_labels == cluster_label)[0]
        kmeans_recommendations = np.random.choice(cluster_indices, size=n, replace=False)
        recommendations["KMeans"] = kmeans_recommendations

        # PCA recommendation
        distances_pca = np.linalg.norm(self.X_pca - self.X_pca[song_index], axis=1)
        pca_recommendations = distances_pca.argsort()[1:n+1]  # Exclude the seed song itself
        recommendations["PCA"] = pca_recommendations

        # t-SNE recommendation
        distances_tsne = np.linalg.norm(self.X_tsne - self.X_tsne[song_index], axis=1)
        tsne_recommendations = distances_tsne.argsort()[1:n+1]  # Exclude the seed song itself
        recommendations["t-SNE"] = tsne_recommendations

        print("Chosen Song", self.data.iloc[song_index]["track_name"])
        for similarity_name, recommended_song_indices in recommendations.items():
            print(f"\nUsing {similarity_name}:")
            recommended_songs = self.data.iloc[recommended_song_indices]
            print(recommended_songs['track_name'])

        return recommendations

    def get_top_recommendations(self, song_name, n=10):
        print(song_name)
        song_index = self.get_song_index_by_name(song_name)
        print(song_index)
        recommendations = {}
        print(self.similarity_scores)
        if self.similarity_scores is None or self.kmeans_labels is None or self.X_pca is None or self.X_tsne is None:
            print("Error: Model attributes are not populated.")
            return None

        for similarity_name, similarity_scores in self.similarity_scores.items():
            try:
                scores = similarity_scores[song_index]
                similar_song_indices = scores.argsort()
                similar_song_indices = similar_song_indices[similar_song_indices != song_index]
                recommendations[similarity_name] = similar_song_indices[:n]
            except Exception as e:
                print(f"An error occurred while loading the model: {e}")
                return None

        # KMeans recommendation
        cluster_label = self.kmeans_labels[song_index]
        cluster_indices = np.where(self.kmeans_labels == cluster_label)[0]
        kmeans_recommendations = np.random.choice(cluster_indices, size=n, replace=False)
        recommendations["KMeans"] = kmeans_recommendations

        # PCA recommendation
        distances_pca = np.linalg.norm(self.X_pca - self.X_pca[song_index], axis=1)
        pca_recommendations = distances_pca.argsort()[1:n+1]  # Exclude the seed song itself
        recommendations["PCA"] = pca_recommendations

        # t-SNE recommendation
        distances_tsne = np.linalg.norm(self.X_tsne - self.X_tsne[song_index], axis=1)
        tsne_recommendations = distances_tsne.argsort()[1:n+1]  # Exclude the seed song itself
        recommendations["t-SNE"] = tsne_recommendations

        counter = Counter()
        for key, value in recommendations.items():
            counter.update(value)

        sorted_counter = counter.most_common()

        new_dataset = {
            'value': [x[0] for x in sorted_counter],
            'count': [x[1] for x in sorted_counter]
        }

        first_10_values = new_dataset['value'][:10]
        first_10_counts = new_dataset['count'][:10]

        first_10_dataset = {
            'value': first_10_values,
            'count': first_10_counts
        }
        top_recommendations = [{}]
        
        print("Chosen Song:", song_name)
        for index in first_10_dataset['value']:
            recommended_song = self.data.iloc[index]
            top_recommendations.append({recommended_song['track_name']: recommended_song['artist_name']})
            print("Song:", recommended_song['track_name'])
            print("Artist:", recommended_song['artist_name'])
            print()  # Add a newline for better readability between recommendations

        return top_recommendations

    def generate_plots(self):
        
        # Timeline of Music by Release Year
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.raw_data, x='release_year', bins=30, kde=False)
        plt.title('Timeline of Music by Release Year')
        plt.xlabel('Release Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
        
        plt.figure(figsize=(16, 12))
        for i, feature in enumerate(self.features):
            plt.subplot(4, 4, i + 1)
            sns.histplot(data=self.data.drop(columns=[x for x in self.data.columns if x != feature]), x=feature, kde=True)
            if feature=='key' or feature=='mode' or feature=='time_signature' :
                plt.title(f'{feature} Distribution')
            else:
                plt.title(f'{feature} Distribution (Scaled)')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.data[self.features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        self.data.boxplot(rot=45)
        plt.title('Boxplots of Numerical Features')
        plt.show()
        
        # Clustering Analysis
        plt.figure(figsize=(12, 6))
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.kmeans_labels, cmap='viridis', alpha=0.5)
        plt.title('PCA Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(self.X_tsne[:, 0], self.X_tsne[:, 1], c=self.kmeans_labels, cmap='viridis', alpha=0.5)
        plt.title('t-SNE Clusters')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(label='Cluster')
        plt.show()
        
        # Model Evaluation
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_pca)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.xticks(range(1, 11))
        plt.grid(True)
        plt.show()
        
        pca = PCA().fit(self.data[self.features])
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance Ratio for PCA')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    model = MusicRecommender()
    if os.path.exists('recommender_model.pkl'):
        print("Model file found. Loading the model...")
        model = MusicRecommender.load_model('recommender_model.pkl')
    else:
        print("Model file not found. Fitting the model...")
        model.fit('top_tracks_data_new.csv')
        model.save_model('recommender_model.pkl')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load data
column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', names=column_names, engine='python')

# Create a user-project matrix
user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

# Convert to numpy array for clustering
user_item_matrix_np = user_item_matrix.to_numpy()

# Divide users into three groups
kmeans = KMeans(n_clusters=3, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix_np)

# Implement group recommendation algorithm
def calculate_recommendations(user_item_matrix, user_clusters, num_recommendations=10):
    group_recommendations = {}
    for group_id in range(3):
        group_indices = np.where(user_clusters == group_id)[0]
        group_matrix = user_item_matrix[group_indices, :]

        # Calculate the average
        avg_scores = group_matrix.mean(axis=0)
        top_avg_recommendations = avg_scores.argsort()[::-1][:num_recommendations]

        # Computational additional utilitarianism
        au_scores = group_matrix.sum(axis=0)
        top_au_recommendations = au_scores.argsort()[::-1][:num_recommendations]

        # Simple counting
        sc_scores = (group_matrix > 0).sum(axis=0)
        top_sc_recommendations = sc_scores.argsort()[::-1][:num_recommendations]

        # Approve the vote
        av_scores = (group_matrix >= 4).sum(axis=0)
        top_av_recommendations = av_scores.argsort()[::-1][:num_recommendations]

        # Borda count
        borda_scores = np.zeros(group_matrix.shape[1])
        for user_ratings in group_matrix:
            ranked_indices = np.argsort(user_ratings)[::-1]
            for rank, idx in enumerate(ranked_indices):
                borda_scores[idx] += (group_matrix.shape[1] - rank - 1)
        top_borda_recommendations = borda_scores.argsort()[::-1][:num_recommendations]

        # Copeland rule
        copeland_scores = np.zeros(group_matrix.shape[1])
        for i in range(group_matrix.shape[1]):
            for j in range(group_matrix.shape[1]):
                if i != j:
                    wins = np.sum(group_matrix[:, i] > group_matrix[:, j])
                    losses = np.sum(group_matrix[:, i] < group_matrix[:, j])
                    copeland_scores[i] += (wins - losses)
        top_copeland_recommendations = copeland_scores.argsort()[::-1][:num_recommendations]

        group_recommendations[group_id] = {
            'Average': top_avg_recommendations,
            'Additive Utilitarian': top_au_recommendations,
            'Simple Count': top_sc_recommendations,
            'Approval Voting': top_av_recommendations,
            'Borda Count': top_borda_recommendations,
            'Copeland Rule': top_copeland_recommendations,
        }
    return group_recommendations

# Calculate the recommendation result
recommendations = calculate_recommendations(user_item_matrix_np, user_clusters)

# Print recommendation results
for group_id, recs in recommendations.items():
    print(f"Group {group_id}:")
    for method, top_items in recs.items():
        print(f"  {method}: {top_items}")

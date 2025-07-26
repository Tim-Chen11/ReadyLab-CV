import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Load data
df = pd.read_csv('run.csv')

# Check package versions
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("PIL version:", PIL.__version__)

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet18 and remove final layer
resnet = models.resnet18(pretrained=True)
encoder = nn.Sequential(*list(resnet.children())[:-1])
encoder.eval()

# Extract feature vector from image
def extract_vector(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(tensor).squeeze().numpy()
    return features

# Apply feature extraction
df['vector'] = df['image_path'].apply(extract_vector)

# Cluster by decade using K-Means
clustered_frames = []
for decade in df['decade'].unique():
    subset = df[df['decade'] == decade].copy()
    vectors = np.stack(subset['vector'].values)
    linkage_matrix = linkage(vectors, method='ward')
    best_k, best_score = 2, -1
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k, best_score = k, score
    final_kmeans = KMeans(n_clusters=5, random_state=42).fit(vectors)
    subset['cluster'] = final_kmeans.labels_
    clustered_frames.append(subset)

# Combine and save clustered data
final_df = pd.concat(clustered_frames)
final_df.to_csv('clustered_products2.csv', index=False)

# Set seaborn style
sns.set(style="whitegrid")

# Define PCA visualization function
def plot_decade_clusters(final_df, decade):
    data = final_df[final_df['decade'] == decade]
    vectors = np.stack(data['vector'].values)
    labels = data['cluster'].values
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(set(labels)))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette=palette)
    plt.title(f"{decade} Product Clusters (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

# Generate scatterplot for popularity
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=popularity_df,
    x='decade',
    y='popularity_score',
    size='descendant_count',
    hue='decade',
    legend='full',
    sizes=(50, 400)
)
plt.title("Design Popularity by Cluster (Based on Descendants)", fontsize=14)
plt.xlabel("Decade")
plt.ylabel("Popularity Score")
plt.tight_layout()
plt.show()

# Export results
similarity_df.to_csv("cluster_similarity_matrix2.csv", index=False)
popularity_df.to_csv("cluster_popularity_rankings2.csv", index=False)
# import pandas as pd
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from scipy.cluster.hierarchy import linkage
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import seaborn as sns

# # Load data
# df = pd.read_csv('run.csv')

# # Check package versions
# print("Torch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("PIL version:", PIL.__version__)

# # Define image transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load pre-trained ResNet18 and remove final layer
# resnet = models.resnet18(pretrained=True)
# encoder = nn.Sequential(*list(resnet.children())[:-1])
# encoder.eval()

# # Extract feature vector from image
# def extract_vector(image_path):
#     image = Image.open(image_path).convert('RGB')
#     tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         features = encoder(tensor).squeeze().numpy()
#     return features

# # Apply feature extraction
# df['vector'] = df['image_path'].apply(extract_vector)

# # Cluster by decade using K-Means
# clustered_frames = []
# for decade in df['decade'].unique():
#     subset = df[df['decade'] == decade].copy()
#     vectors = np.stack(subset['vector'].values)
#     linkage_matrix = linkage(vectors, method='ward')
#     best_k, best_score = 2, -1
#     for k in range(2, 6):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         labels = kmeans.fit_predict(vectors)
#         score = silhouette_score(vectors, labels)
#         if score > best_score:
#             best_k, best_score = k, score
#     final_kmeans = KMeans(n_clusters=5, random_state=42).fit(vectors)
#     subset['cluster'] = final_kmeans.labels_
#     clustered_frames.append(subset)

# # Combine and save clustered data
# final_df = pd.concat(clustered_frames)
# final_df.to_csv('clustered_products2.csv', index=False)

# # Set seaborn style
# sns.set(style="whitegrid")

# # Define PCA visualization function
# def plot_decade_clusters(final_df, decade):
#     data = final_df[final_df['decade'] == decade]
#     vectors = np.stack(data['vector'].values)
#     labels = data['cluster'].values
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(vectors)
#     plt.figure(figsize=(10, 6))
#     palette = sns.color_palette("husl", len(set(labels)))
#     sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette=palette)
#     plt.title(f"{decade} Product Clusters (PCA)")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.legend(title="Cluster")
#     plt.show()

# # Generate scatterplot for popularity
# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     data=popularity_df,
#     x='decade',
#     y='popularity_score',
#     size='descendant_count',
#     hue='decade',
#     legend='full',
#     sizes=(50, 400)
# )
# plt.title("Design Popularity by Cluster (Based on Descendants)", fontsize=14)
# plt.xlabel("Decade")
# plt.ylabel("Popularity Score")
# plt.tight_layout()
# plt.show()

# # Export results
# similarity_df.to_csv("cluster_similarity_matrix2.csv", index=False)
# popularity_df.to_csv("cluster_popularity_rankings2.csv", index=False)






import json
import hashlib
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

def setup_feature_extractor():
    """Initialize the ResNet18 feature extractor"""
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
    
    return transform, encoder

def get_image_path_from_url(url: str, image_cache_dir: Path) -> Path:
    """Generate image path from URL using the same hashing method"""
    filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
    return image_cache_dir / filename

def extract_features_batch(image_paths: List[Path], transform, encoder, batch_size: int = 32) -> List[np.ndarray]:
    """Extract features in batches for better GPU utilization"""
    features_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        valid_indices = []
        
        # Load and preprocess batch
        for j, path in enumerate(batch_paths):
            try:
                image = Image.open(path).convert('RGB')
                tensor = transform(image)
                batch_tensors.append(tensor)
                valid_indices.append(i + j)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                features_list.append(None)
                continue
        
        if not batch_tensors:
            continue
            
        # Process batch
        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_features = encoder(batch_tensor).squeeze().cpu().numpy()
        
        # Handle single image case
        if len(batch_tensors) == 1:
            batch_features = batch_features.reshape(1, -1)
            
        # Add to results
        batch_idx = 0
        for j in range(len(batch_paths)):
            if i + j in valid_indices:
                features_list.append(batch_features[batch_idx])
                batch_idx += 1
            else:
                features_list.append(None)
    
    return features_list

def find_optimal_clusters(vectors: np.ndarray, min_k: int = 2, max_k: int = 6) -> int:
    """Find optimal number of clusters using silhouette score"""
    if len(vectors) < min_k:
        return min(len(vectors), 2)
    
    best_k, best_score = min_k, -1
    
    for k in range(min_k, min(max_k + 1, len(vectors) + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # Silhouette score requires at least 2 clusters and 2 samples
        if len(set(labels)) > 1:
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_k, best_score = k, score
    
    return best_k

def cluster_images_by_decade(metadata: List[Dict[Any, Any]], image_cache_dir: Path, 
                           batch_size: int = 32, use_fixed_k: bool = False, fixed_k: int = 5) -> List[Dict[Any, Any]]:
    """
    Cluster images by decade and add cluster information to metadata
    
    Args:
        metadata: List of metadata dictionaries
        image_cache_dir: Path to cached images directory
        batch_size: Batch size for feature extraction
        use_fixed_k: Whether to use fixed number of clusters (faster)
        fixed_k: Fixed number of clusters if use_fixed_k=True
    
    Returns:
        Updated metadata with cluster information
    """
    print("Setting up feature extractor...")
    transform, encoder = setup_feature_extractor()
    
    # Group metadata by decade using defaultdict for efficiency
    decade_groups = defaultdict(list)
    for item in metadata:
        decade_groups[item['decade']].append(item)
    
    updated_metadata = []
    
    for decade, items in decade_groups.items():
        print(f"\nProcessing decade: {decade} ({len(items)} images)")
        
        # Prepare image paths and check existence upfront
        valid_items = []
        image_paths = []
        
        for item in items:
            image_path = get_image_path_from_url(item['url'], image_cache_dir)
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                item['cluster'] = -1
                updated_metadata.append(item)
                continue
            
            valid_items.append(item)
            image_paths.append(image_path)
        
        if len(image_paths) == 0:
            print(f"No valid images found for decade {decade}")
            continue
        
        if len(image_paths) == 1:
            valid_items[0]['cluster'] = 0
            updated_metadata.extend(valid_items)
            continue
        
        # Extract features in batches
        print(f"Extracting features for {len(image_paths)} images...")
        features_list = extract_features_batch(image_paths, transform, encoder, batch_size)
        
        # Filter out failed extractions
        final_items = []
        final_vectors = []
        
        for item, features in zip(valid_items, features_list):
            if features is not None:
                final_items.append(item)
                final_vectors.append(features)
            else:
                item['cluster'] = -1
                updated_metadata.append(item)
        
        if len(final_vectors) == 0:
            print(f"No valid features extracted for decade {decade}")
            continue
        
        if len(final_vectors) == 1:
            final_items[0]['cluster'] = 0
            updated_metadata.extend(final_items)
            continue
        
        # Convert to numpy array
        vectors = np.array(final_vectors)
        print(f"Feature extraction complete. Shape: {vectors.shape}")
        
        # Determine number of clusters
        if use_fixed_k:
            k = min(fixed_k, len(vectors))
            print(f"Using fixed k={k}")
        else:
            k = find_optimal_clusters(vectors)
            print(f"Optimal number of clusters: {k}")
        
        # Perform clustering
        if k > 1:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init for speed
            cluster_labels = kmeans.fit_predict(vectors)
        else:
            cluster_labels = np.zeros(len(vectors), dtype=int)
        
        # Add cluster information to metadata
        for item, cluster_label in zip(final_items, cluster_labels):
            item['cluster'] = int(cluster_label)
        
        updated_metadata.extend(final_items)
        
        print(f"Clustering complete. Cluster distribution: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    
    return updated_metadata

def add_clustering_to_pipeline(data_dir: Path, batch_size: int = 32, use_fixed_k: bool = False, fixed_k: int = 5):
    """
    Add clustering step to the existing data processing pipeline
    
    Args:
        data_dir: Data directory path
        batch_size: Batch size for feature extraction (larger = faster but more memory)
        use_fixed_k: Use fixed number of clusters instead of optimization (much faster)
        fixed_k: Number of clusters to use if use_fixed_k=True
    """
    processed_json_path = data_dir / 'metadata' / 'processed_metadata.json'
    image_cache_dir = data_dir / 'cache' / 'images'
    
    # Load existing processed metadata
    print("Loading processed metadata...")
    with open(processed_json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} items from metadata")
    
    # Check GPU availability
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device}")
    
    # Perform clustering
    print("Starting clustering process...")
    clustered_metadata = cluster_images_by_decade(
        metadata, image_cache_dir, batch_size, use_fixed_k, fixed_k
    )
    
    # Save updated metadata
    print(f"Saving clustered metadata back to {processed_json_path}")
    
    with open(processed_json_path, 'w', encoding='utf-8') as f:
        json.dump(clustered_metadata, f, indent=2, ensure_ascii=False)
    
    print("Clustering complete!")
    
    # Print summary statistics
    cluster_stats = {}
    missing_images = 0
    failed_extractions = 0
    
    for item in clustered_metadata:
        decade = item['decade']
        cluster = item.get('cluster', -1)
        
        if cluster == -1:
            if 'cluster' in item:
                failed_extractions += 1
            else:
                missing_images += 1
            continue
            
        if decade not in cluster_stats:
            cluster_stats[decade] = {}
        
        if cluster not in cluster_stats[decade]:
            cluster_stats[decade][cluster] = 0
        cluster_stats[decade][cluster] += 1
    
    print("\n=== Clustering Summary ===")
    for decade, clusters in cluster_stats.items():
        print(f"{decade}: {len(clusters)} clusters, {sum(clusters.values())} images")
        for cluster_id, count in sorted(clusters.items()):
            print(f"  Cluster {cluster_id}: {count} images")
    
    if missing_images > 0:
        print(f"\nWarning: {missing_images} images were not found in cache")
    if failed_extractions > 0:
        print(f"Warning: {failed_extractions} images failed feature extraction")
    
    return processed_json_path
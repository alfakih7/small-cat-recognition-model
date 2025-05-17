import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
import random

# Load the CSV file
csv_path = '/kaggle/input/tammathon-task-1/train.csv'
print(f"Loading CSV from: {csv_path}")

# Load the CSV and filter to include only the first 10,000 classes
df = pd.read_csv(csv_path)
print(f"CSV loaded with {len(df)} total entries")

# Filter to only include first 10,000 classes
df['label'] = df['label'].astype(int)  # Ensure label is integer
filtered_df = df[df['label'] < 10000]
print(f"Filtered to {len(filtered_df)} entries with labels < 10000")

# Save filtered CSV temporarily
filtered_csv_path = '/kaggle/working/filtered_train.csv'
filtered_df.to_csv(filtered_csv_path, index=False)

# Create a dataset for the filtered CSV images
class CSVImageDataset(Dataset):
    def __init__(self, csv_file, base_dir='/kaggle/input/tammathon-task-1', transform=None):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform
        
        # Clean up and validate data
        self.image_paths = []
        self.labels = []
        self.filenames = []
        
        for idx, row in self.df.iterrows():
            try:
                filename = row['filename']
                label = row['label']
                
                # Construct full path (try different possible paths)
                img_path = None
                possible_paths = [
                    os.path.join(self.base_dir, filename),
                    os.path.join(self.base_dir, 'train', filename),
                    filename if os.path.isabs(filename) else os.path.join(self.base_dir, filename)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break
                
                if img_path:
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.filenames.append(filename)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        print(f"Found {len(self.image_paths)} valid images out of {len(self.df)} entries")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            img = self.transform(img)
        
        return img, idx, str(label)  # Return as (image, index, label)

# Create dataset and dataloader for filtered CSV images
csv_dataset = CSVImageDataset(filtered_csv_path, transform=val_transform)
csv_loader = DataLoader(csv_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load the best model
print("Loading best model...")
best_model_path = 'best_model_phase2.pth'  # Use Phase 2 model if it's better, otherwise use Phase 1
if not os.path.exists(best_model_path):
    best_model_path = 'best_model_phase1.pth'
    
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()

# Extract embeddings for filtered CSV images
print("Extracting embeddings for CSV images (first 10,000 classes only)...")
csv_embeddings, csv_indices, csv_labels = extract_embeddings_triplet(model, csv_loader, device)
print(f"Extracted {len(csv_embeddings)} embeddings")

# Normalize embeddings
csv_embeddings = csv_embeddings.astype(np.float32)
faiss.normalize_L2(csv_embeddings)

# Create index for searching
print("Creating FAISS index...")
dimension = csv_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(csv_embeddings)

# Evaluate model predictions
print("Evaluating model predictions...")
correct_predictions = 0
total_predictions = 0
correct_examples = []

# Group images by label
label_to_indices = {}
for i, label in enumerate(csv_labels):
    if label not in label_to_indices:
        label_to_indices[label] = []
    label_to_indices[label].append(i)

# Evaluate each image
for idx, label in enumerate(tqdm(csv_labels, desc="Evaluating")):
    # Skip if this label has only one image
    if len(label_to_indices[label]) <= 1:
        continue
    
    # Use current image as query
    query_embedding = csv_embeddings[idx:idx+1]
    query_path = csv_dataset.image_paths[idx]
    query_label = label
    
    # Search for similar images (k=2 to include the query image itself)
    D, I = index.search(query_embedding, k=2)
    
    # Get top match (excluding self)
    top_match_idx = I[0][1] if I[0][0] == idx else I[0][0]
    predicted_label = csv_labels[top_match_idx]
    
    # Check if prediction is correct
    is_correct = (predicted_label == query_label)
    
    if is_correct:
        correct_predictions += 1
        # Store example if it's correct
        correct_examples.append({
            'query_path': query_path,
            'query_label': query_label,
            'pred_path': csv_dataset.image_paths[top_match_idx],
            'pred_label': predicted_label,
            'score': D[0][1] if I[0][0] == idx else D[0][0]
        })
    
    total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"\nModel Performance Overview (First 10,000 Classes Only):")
print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Display 10 examples of correct predictions
if correct_examples:
    # Sort by confidence (score)
    correct_examples.sort(key=lambda x: x['score'], reverse=True)
    
    # Select 10 random examples from the top 100 to show diversity
    top_examples = correct_examples[:100]
    examples_to_show = random.sample(top_examples, min(10, len(top_examples)))
    
    # Create figure
    fig, axes = plt.subplots(10, 2, figsize=(12, 25))
    fig.suptitle('Examples of Correct Predictions (Query: Prediction)', fontsize=16)
    
    for i, example in enumerate(examples_to_show):
        if i >= 10:
            break
            
        # Display query image
        try:
            query_img = Image.open(example['query_path']).convert('RGB')
            axes[i, 0].imshow(query_img)
            axes[i, 0].set_title(f"Query: Label {example['query_label']}")
            axes[i, 0].axis('off')
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, "Error loading image", ha='center', va='center')
            axes[i, 0].axis('off')
        
        # Display predicted image
        try:
            pred_img = Image.open(example['pred_path']).convert('RGB')
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f"Prediction: Label {example['pred_label']}\nScore: {example['score']:.3f}")
            axes[i, 1].axis('off')
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, "Error loading image", ha='center', va='center')
            axes[i, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
else:
    print("No correct predictions to display.")

print("Evaluation complete!")
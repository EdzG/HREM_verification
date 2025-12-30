import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Configuration
RESULTS_PATH = 'runs/f30k_test/results_f30k.npy'
IMAGE_DIR = 'data/f30k/images/' 
OUTPUT_PATH = 'runs/f30k_test/retrieval_viz.png'
QUERY_CAPTION_IDX = 0  
TOP_K = 5

def load_visuals():
    # 2. Load and Unpack Similarity Matrix
    try:
        data = np.load(RESULTS_PATH, allow_pickle=True)
        sims = data.item() if data.ndim == 0 else data
        if isinstance(sims, dict):
            sims = sims.get('s', sims.get('sims', sims))
        print(f"Loaded matrix. Shape: {sims.shape}")
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return

    # 3. Robust Filename Mapping
    # We first try to find the standard Flickr30k test split file
    image_filenames = []
    # common locations for the test split file in HREM/VSE structures
    split_files = ['data/f30k/test.txt', 'data/f30k/test_ids.txt', 'data/f30k/test_filenames.txt']
    
    for split_file in split_files:
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                # Extract the first column (the ID/filename)
                image_filenames = [line.strip().split()[0] for line in f.readlines()]
                print(f"Found mapping file: {split_file}")
                break

    # 4. Emergency Fallback: Scan the actual image directory
    if not image_filenames:
        print("Mapping file not found. Scanning image directory as fallback...")
        all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg'))])
        if len(all_files) >= sims.shape[0]:
            image_filenames = all_files[:sims.shape[0]]
            print(f"Fallback: Using first {len(image_filenames)} files found in {IMAGE_DIR}")
        else:
            print(f"Critical Error: Only {len(all_files)} images found, but matrix needs {sims.shape[0]}.")
            return

    # 5. Find Top-K Matches
    # Image-to-Text retrieval logic
    caption_scores = sims[:, QUERY_CAPTION_IDX] 
    top_indices = np.argsort(caption_scores)[::-1][:TOP_K]

    # 6. Generate Visualization
    plt.figure(figsize=(20, 10))
    print(f"Visualizing Top {TOP_K} matches...")

    for i, img_idx in enumerate(top_indices):
        img_name = image_filenames[img_idx]
        if not img_name.lower().endswith(('.jpg', '.jpeg')):
            img_name += '.jpg'
            
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        plt.subplot(1, TOP_K, i + 1)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Rank {i+1}\n{img_name}\nScore: {caption_scores[img_idx]:.4f}")
        else:
            # If still missing, show the path it tried to find
            plt.text(0.5, 0.5, f"FILE NOT FOUND:\n{img_name}\n\nChecked in:\n{IMAGE_DIR}", 
                     ha='center', va='center', color='red', fontsize=8)
            plt.title(f"Rank {i+1} (Error)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Visualization saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    load_visuals()
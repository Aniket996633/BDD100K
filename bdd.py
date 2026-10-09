import os
import yaml
import glob
import cv2
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import FileLink

# =====================================================================
# 1. SETUP ENVIRONMENT & CONFIGURATION
# =====================================================================
DATASET_PATH = "/kaggle/input/datasets/a7madmostafa/bdd100k-yolo"
WORKING_DIR = "/kaggle/working/bdd100k_pipeline"
os.makedirs(WORKING_DIR, exist_ok=True)

print("Initializing BDD100K Object Detection Pipeline Context...")

# Mapping Strategy to isolate the 5 road-user classes
CLASS_MAPPING = {
    2: 0,  # car -> car
    3: 1,  # bus -> bus
    4: 2,  # truck -> truck
    0: 3,  # person -> pedestrian
    1: 4,  # rider -> cyclist
    5: 4   # bike -> cyclist
}

# =====================================================================
# 2. DATA PROCESSING & ENVIRONMENT PREPARATION
# =====================================================================
def preprocess_labels_and_slice_metadata(source_dir, output_dir, split="train"):
    src_labels_path = os.path.join(source_dir, split, "labels")
    dst_labels_path = os.path.join(output_dir, split, "labels")
    dst_images_path = os.path.join(output_dir, split, "images")
    
    os.makedirs(dst_labels_path, exist_ok=True)
    os.makedirs(dst_images_path, exist_ok=True)
    
    label_files = glob.glob(os.path.join(src_labels_path, "*.txt"))
    
    # Process a representative subset to optimize notebook runtime constraints
    max_images = 12000 if split == "train" else 2000
    for count, label_file in enumerate(label_files[:max_images]):
        file_id = Path(label_file).stem
        src_img_file = os.path.join(source_dir, split, "images", f"{file_id}.jpg")
        
        if not os.path.exists(src_img_file):
            continue
            
        valid_boxes = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            orig_cls = int(parts[0])
            
            if orig_cls in CLASS_MAPPING:
                new_cls = CLASS_MAPPING[orig_cls]
                bbox_coords = " ".join(parts[1:])
                valid_boxes.append(f"{new_cls} {bbox_coords}\n")
        
        dst_lbl_file = os.path.join(dst_labels_path, f"{file_id}.txt")
        with open(dst_lbl_file, 'w') as f:
            f.writelines(valid_boxes)
            
        dst_img_file = os.path.join(dst_images_path, f"{file_id}.jpg")
        if not os.path.exists(dst_img_file):
            os.symlink(src_img_file, dst_img_file)

# Build configuration structure
preprocess_labels_and_slice_metadata(DATASET_PATH, WORKING_DIR, "train")
preprocess_labels_and_slice_metadata(DATASET_PATH, WORKING_DIR, "val")

yaml_content = {
    'path': WORKING_DIR,
    'train': 'train/images',
    'val': 'val/images',
    'nc': 5,
    'names': ['car', 'bus', 'truck', 'pedestrian', 'cyclist']
}

yaml_file_path = os.path.join(WORKING_DIR, 'bdd100k_5classes.yaml')
with open(yaml_file_path, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

# =====================================================================
# 3. ENVIRONMENT IMPORT & DEPLOYMENT RUN
# =====================================================================
!pip install -q ultralytics
from ultralytics import YOLO

print("Starting pipeline training run...")
improved_model = YOLO('yolov8n.pt')

# Fine-tuned for small vehicle detection profiles
improved_results = improved_model.train(
    data=yaml_file_path,
    epochs=10,
    imgsz=640,
    batch=32,
    device=0,
    box=10.5,             
    cls=2.0,              
    scale=0.6,            
    workers=4,
    name='bdd100k_improved'
)

# Validate results
print("Generating final metrics and evaluation plots...")
improved_metrics = improved_model.val(name='eval_improved')

# =====================================================================
# 4. FAIL-SAFE COMPRESSION & PERSISTENT FILE DOWNLOAD
# =====================================================================
print("Securing final metrics for download...")
target_eval_dir = '/kaggle/working/runs/detect/eval_improved'
zip_output_path = '/kaggle/working/bdd100k_final_results'

if os.path.exists(target_eval_dir):
    # Pack charts, performance profiles, and predictions into a single zip file
    shutil.make_archive(zip_output_path, 'zip', target_eval_dir)
    print("=" * 60)
    print("🎉 PIPELINE COMPLETE AND SECURED!")
    print("=" * 60)
    # Generate interactive download link directly inside the cell output
    display(FileLink(f"{zip_output_path}.zip", result_html_prefix="Click here to download your project deliverables: "))
else:
    print("⚠️ Evaluation files not found in the primary directory path.")

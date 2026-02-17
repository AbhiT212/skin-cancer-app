import torch
import numpy as np
import torchvision.transforms as transforms
from config import CONFIG, SEX_COLS, LOC_COLS

def process_metadata(age, sex, localization):
    """Accurately maps inputs to the 19-dim vector expected by the model."""
    meta_vector = np.zeros(CONFIG['num_meta_features'], dtype=np.float32)
    
    # Age Scaling (mean and std from HAM10000 dataset)
    mean_age, std_age = 51.86, 16.96 
    meta_vector[0] = (age - mean_age) / std_age
    
    # Sex One-Hot
    sex_col = f"sex_{sex.lower()}"
    if sex_col in SEX_COLS:
        idx = 1 + SEX_COLS.index(sex_col)
        meta_vector[idx] = 1.0
        
    # Localization One-Hot
    loc_col = f"localization_{localization.lower()}"
    if loc_col in LOC_COLS:
        idx = 1 + len(SEX_COLS) + LOC_COLS.index(loc_col)
        meta_vector[idx] = 1.0
        
    return torch.tensor(meta_vector, dtype=torch.float32)

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
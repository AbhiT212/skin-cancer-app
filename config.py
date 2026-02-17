import torch

CONFIG = {
    'img_size': 300,
    'num_classes': 7,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_meta_features': 19 
}

LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis',
    'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions', 'df': 'Dermatofibroma'
}

IDX_TO_CLASS = {i: name for i, name in enumerate(LESION_TYPE_DICT.values())}

# Standard HAM10000 Metadata Columns
SEX_COLS = ['sex_female', 'sex_male', 'sex_unknown']
LOC_COLS = [
    'localization_abdomen', 'localization_acral', 'localization_back', 
    'localization_chest', 'localization_ear', 'localization_face', 
    'localization_foot', 'localization_genital', 'localization_hand', 
    'localization_lower extremity', 'localization_neck', 'localization_scalp', 
    'localization_trunk', 'localization_unknown', 'localization_upper extremity'
]
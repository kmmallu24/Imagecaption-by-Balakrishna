# Paths for data
image_paths = {
    "train": "/content/drive/MyDrive/satiliteimagecaption/train",
    "valid": "/content/drive/MyDrive/satiliteimagecaption/valid",
    "test": "/content/drive/MyDrive/satiliteimagecaption/test",
}
caption_paths = {
    "train": "/content/drive/MyDrive/satiliteimagecaption/train.csv",
    "valid": "/content/drive/MyDrive/satiliteimagecaption/valid.csv",
    "test": "/content/drive/MyDrive/satiliteimagecaption/test.csv",
}
output_path = "/content/drive/MyDrive/satiliteimagecaption/preprocessed_data4"

# Load Vision Transformer Model and Feature Extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/vit-base-patch16-224"
vit_model = ViTModel.from_pretrained(model_name, use_auth_token=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, use_auth_token=True)

vit_model = vit_model.to(device).eval()

# Create output directory if not exists
os.makedirs(output_path, exist_ok=True)
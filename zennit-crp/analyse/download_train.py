import kagglehub

# Download latest version
path = kagglehub.dataset_download("sautkin/imagenet1k1")

print("Path to dataset files:", path)
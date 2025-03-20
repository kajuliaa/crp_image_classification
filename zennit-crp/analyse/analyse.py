import tarfile
from collections import Counter

# Path to the ImageNet tar.gz file
imagenet_tar_path = "/home/kaidashova/data/ImageNet_data/ILSVRC2012_img_val.tar"

# Dictionary to store class counts
class_counts = Counter()

# Open the tar file without extracting
with tarfile.open(imagenet_tar_path, "r") as tar:
    for member in tar.getmembers():
        if member.isfile():  # Only count files, ignore directories
            # Extract class name from the file path (assumes format "train/class_name/image.jpg")
            parts = member.name.split("/")
            if len(parts) > 1:  # Ensure valid structure
                class_name = parts[-2]  # Class name is usually the parent folder
                class_counts[class_name] += 1

# Convert to sorted list
class_distribution = sorted(class_counts.values(), reverse=True)

# Compute imbalance ratio
imbalance_ratio = max(class_distribution) / min(class_distribution)

print(f"Number of classes: {len(class_counts)}")
print(f"Max images in a class: {max(class_distribution)}")
print(f"Min images in a class: {min(class_distribution)}")
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

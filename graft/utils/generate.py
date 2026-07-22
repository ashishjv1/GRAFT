import os
from medmnist import DermaMNIST
from PIL import Image

# Initialize the dataset
train_dataset = DermaMNIST(split='train', download=True, size=224)
valid_dataset = DermaMNIST(split='val', download=True, size=224)
test_dataset = DermaMNIST(split='test', download=True, size=224)

# Define the root directory for the reorganized dataset
root_dir = 'DermaMNIST'

# Define the subdirectories
subdirs = ['train', 'valid', 'test']
classes = [str(i) for i in range(7)]  # Assuming class labels are 0 through 6

# Create directories
for subdir in subdirs:
    for cls in classes:
        os.makedirs(os.path.join(root_dir, subdir, cls), exist_ok=True)

def save_images(dataset, subdir):
    for idx, (img, label) in enumerate(dataset):
        label = str(int(label[0]))  # Convert label to int and then to string
        img_path = os.path.join(root_dir, subdir, label, f"{subdir}_{idx}.png")
        img.save(img_path)

# Save images to corresponding directories
save_images(train_dataset, 'train')
save_images(valid_dataset, 'valid')
save_images(test_dataset, 'test')

print("Dataset reorganized successfully.")

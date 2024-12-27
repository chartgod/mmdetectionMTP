import os
from PIL import Image

def crop_image(image_path, output_dir, crop_size=256):
    """
    Crop a large image into smaller patches of specified size.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory where cropped images will be saved.
        crop_size (int): Size of the cropped patches.
    """
    img = Image.open(image_path)
    img_width, img_height = img.size

    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, img_width, crop_size):
        for j in range(0, img_height, crop_size):
            box = (i, j, i + crop_size, j + crop_size)
            cropped_img = img.crop(box)
            cropped_img.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.png"))

# Input and output directories
input_dir = r"/home/lsh/share/CD/open-cd/data/1218_시나리오테스트"
output_dir = r"/home/lsh/share/CD/open-cd/data/1218_시나리오테스트/crop256/test"

# Ensure the cropped_test directory has the same structure as the test directory
subdirs = ['A','B','label']
for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

# Crop images in the test directory
for subdir in subdirs:
    input_subdir = os.path.join(input_dir, subdir)
    output_subdir = os.path.join(output_dir, subdir)
    for root, dirs, files in os.walk(input_subdir):
        for file in files:
            if file.endswith(".png"):  # Adjust the file extension if needed
                image_path = os.path.join(root, file)
                crop_image(image_path, output_subdir, crop_size=256)

import os
from PIL import Image

def combine_images(image_dir, output_dir, combined_size=1024, crop_size=256):
    """
    Combine smaller image patches into a larger image of specified size.

    Args:
        image_dir (str): Directory containing cropped image patches.
        output_dir (str): Directory where combined images will be saved.
        combined_size (int): Size of the combined images (width and height).
        crop_size (int): Size of the cropped patches.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all cropped images
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    if not images:
        print(f"No images found in {image_dir}.")
        return

    num_patches = combined_size // crop_size
    image_patches = {}

    # Organize patches by their position
    for image_name in images:
        name_parts = image_name.split('_')
        base_name = '_'.join(name_parts[:-2])
        x = int(name_parts[-2])
        y = int(name_parts[-1].split('.')[0])
        
        if base_name not in image_patches:
            image_patches[base_name] = {}
        
        image_patches[base_name][(x, y)] = image_name

    # Combine patches into a larger image
    for base_name, patches in image_patches.items():
        combined_img = Image.new('RGB', (combined_size, combined_size))

        for x in range(0, combined_size, crop_size):
            for y in range(0, combined_size, crop_size):
                patch_name = patches.get((x, y))
                if patch_name:
                    patch_img = Image.open(os.path.join(image_dir, patch_name))
                    combined_img.paste(patch_img, (x, y))

        output_file_name = f"{base_name}.png"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Handle duplicate file names
        duplicate_counter = 1
        while os.path.exists(output_file_path):
            output_file_name = f"{base_name}_{duplicate_counter}.png"
            output_file_path = os.path.join(output_dir, output_file_name)
            duplicate_counter += 1

        if duplicate_counter > 1:
            with open(os.path.join(output_dir, "duplicate_log.txt"), "a") as log_file:
                log_file.write(f"Duplicate: {base_name}.png -> {output_file_name}\n")

        combined_img.save(output_file_path)
        print(f"Saved combined image: {output_file_path}")

# Input and output directories
input_dir = "/home/lsh/share/CD/open-cd/data/10000_total_test/test/label"
output_dir = "/home/lsh/share/CD/open-cd/data/10000_total_test/test/1024"
#input_dir = rf"V:\CD\open-cd\test\총합\시나리오테스트2_epoch150_mtp\vis_data\vis_image"
#output_dir = rf"V:\CD\open-cd\test\총합\시나리오테스트2_epoch150_mtp\1024"

# Combine images in the input directory
combine_images(input_dir, output_dir, combined_size=1024, crop_size=256)
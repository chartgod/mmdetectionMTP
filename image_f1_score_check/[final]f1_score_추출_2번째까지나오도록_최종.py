import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import glob
import shutil

def calculate_f1_score(true_mask, pred_mask):
    true_mask = np.array(true_mask).flatten()
    pred_mask = np.array(pred_mask).flatten()
    
    # Binarize the masks: set all non-zero values to 1
    true_mask = (true_mask > 0).astype(int)
    pred_mask = (pred_mask > 0).astype(int)
    
    return f1_score(true_mask, pred_mask, average='binary')

def process_images(label_dir, predict_dir, output_file):
    label_files = glob.glob(os.path.join(label_dir, "*.png"))
    f1_scores = []

    with open(output_file, 'w') as f:
        f.write("F1 Score Results\n")
        f.write("================\n")

        for label_path in label_files:
            filename = os.path.basename(label_path)
            predict_path = os.path.join(predict_dir, filename)

            if os.path.exists(predict_path):
                try:
                    true_mask = Image.open(label_path).convert('L')
                    pred_mask = Image.open(predict_path).convert('L')

                    # Ensure both images have the same size
                    if true_mask.size != pred_mask.size:
                        pred_mask = pred_mask.resize(true_mask.size, Image.NEAREST)

                    f1 = calculate_f1_score(true_mask, pred_mask)
                    f1_percentage = f1 * 100
                    f1_scores.append((filename, f1_percentage))
                    f.write(f"{filename}: {f1_percentage:.4f}%\n")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

    return f1_scores

def copy_images(filenames, source_dir, target_dir, prefix):
    for idx, filename in enumerate(filenames):
        src = os.path.join(source_dir, filename)
        if os.path.exists(src):
            target_filename = f"{prefix}{idx + 1}_{filename}"
            dst = os.path.join(target_dir, target_filename)
            shutil.copy2(src, dst)

def copy_related_images(base_dir, target_dir, filenames, prefixes):
    for subdir in ['A', 'B', 'label']:
        src_dir = os.path.join(base_dir, subdir)
        dest_dir = os.path.join(target_dir, subdir)
        os.makedirs(dest_dir, exist_ok=True)

        for idx, filename in enumerate(filenames):
            src = os.path.join(src_dir, filename)
            if os.path.exists(src):
                target_filename = f"{prefixes[idx]}{idx + 1}_{filename}"
                shutil.copy2(src, os.path.join(dest_dir, target_filename))

def main(category="수계", predict_type="changer"):
    label_dir = r"D:\국토위성\데이터셋\과년도\Best\{}\label".format(category)
    predict_dir = r"D:\국토위성\데이터셋\과년도\Best\{}\predict\{}".format(category, predict_type)
    output_file = r"D:\국토위성\데이터셋\과년도\Best\f1-score\{}\{}\{}_{}_f1_scores.txt".format(category, predict_type, category, predict_type)
    copy_target_dir_predict = r"D:\국토위성\데이터셋\과년도\Best\f1-score\{}\{}\predict".format(category, predict_type)
    copy_target_dir_related = r"D:\국토위성\데이터셋\과년도\Best\f1-score\{}\{}".format(category, predict_type)

    # Ensure directories exist
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    os.makedirs(copy_target_dir_predict, exist_ok=True)
    os.makedirs(copy_target_dir_related, exist_ok=True)

    # Process images and calculate F1 scores
    f1_scores = process_images(label_dir, predict_dir, output_file)

    if f1_scores:
        non_zero_scores = [(filename, score) for filename, score in f1_scores if score > 0]
        if non_zero_scores:
            sorted_scores = sorted(non_zero_scores, key=lambda x: x[1], reverse=True)
            
            best_images = [filename for filename, score in sorted_scores[:2]]
            
            # Extract the last two (worst) scores and explicitly sort them by F1 score
            worst_images = sorted(sorted_scores[-2:], key=lambda x: x[1])
            
            print(f"Best performing images: {best_images} with F1-scores: {[score for _, score in sorted_scores[:2]]}")
            print(f"Worst performing images: {[filename for filename, _ in worst_images]} with F1-scores: {[score for _, score in worst_images]}")

            # Append summary of best and worst images to the output file
            with open(output_file, 'a') as f:
                f.write("\n--- Summary ---\n")
                for idx, (filename, score) in enumerate(sorted_scores[:2]):
                    f.write(f"Best {idx + 1} performing image: {filename} with F1-score: {score:.4f}%\n")
                for idx, (filename, score) in enumerate(worst_images):
                    f.write(f"Worst {idx + 1} performing image: {filename} with F1-score: {score:.4f}%\n")
            
            # Copy best and worst images from predict to f1-score\category\predict
            copy_images(best_images, predict_dir, copy_target_dir_predict, "best")
            copy_images([filename for filename, _ in worst_images], predict_dir, copy_target_dir_predict, "worst")
            print(f"Copied best and worst images to {copy_target_dir_predict}")

            # Copy corresponding A, B, label images to f1-score\category\A, B, label
            copy_related_images(label_dir, copy_target_dir_related, best_images, ["best1_", "best2_"])
            copy_related_images(label_dir, copy_target_dir_related, [filename for filename, _ in worst_images], ["worst1_", "worst2_"])
            copy_related_images(r"D:\국토위성\데이터셋\과년도\Best\{}".format(category), copy_target_dir_related, best_images, ["best1_", "best2_"])
            copy_related_images(r"D:\국토위성\데이터셋\과년도\Best\{}".format(category), copy_target_dir_related, [filename for filename, _ in worst_images], ["worst1_", "worst2_"])
            print(f"Copied related A, B, label images to {copy_target_dir_related}")
        else:
            print("All non-zero F1-scores are 0.")
    else:
        print("No matching image pairs found or all pairs failed to process.")

if __name__ == "__main__":
    main()  # 기본 설정: category="수계", predict_type="changer"

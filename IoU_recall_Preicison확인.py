import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import glob

def calculate_metrics(true_mask, pred_mask):
    """
    Calculate F1-score, Precision, Recall, and IoU.
    """
    true_mask = np.array(true_mask).flatten()
    pred_mask = np.array(pred_mask).flatten()
    
    # Binarize the masks
    true_mask = (true_mask > 0).astype(int)
    pred_mask = (pred_mask > 0).astype(int)
    
    f1 = f1_score(true_mask, pred_mask, average='binary') * 100
    precision = precision_score(true_mask, pred_mask, average='binary', zero_division=0) * 100
    recall = recall_score(true_mask, pred_mask, average='binary', zero_division=0) * 100
    iou = jaccard_score(true_mask, pred_mask, average='binary') * 100
    
    return f1, precision, recall, iou

def process_images(label_dir, predict_dir, output_file):
    """
    Process images, calculate metrics, and save results to output file.
    """
    label_files = glob.glob(os.path.join(label_dir, "*.[Pp][Nn][Gg]"))
    metrics_data = []

    with open(output_file, 'w') as f:
        f.write("Performance Results (Sorted by IoU)\n")
        f.write("=" * 70 + "\n")
        f.write("Rank\tFilename\tIoU (%)\tF1 (%)\tPrecision (%)\tRecall (%)\n")
        f.write("=" * 70 + "\n")

        for idx, label_path in enumerate(label_files, start=1):
            filename = os.path.basename(label_path)
            predict_path = os.path.join(predict_dir, filename)

            print(f"Processing {idx}/{len(label_files)}: {filename}...")

            if os.path.exists(predict_path):
                try:
                    true_mask = Image.open(label_path).convert('L')
                    pred_mask = Image.open(predict_path).convert('L')

                    # Resize predict mask to match label mask size
                    if true_mask.size != pred_mask.size:
                        pred_mask = pred_mask.resize(true_mask.size, Image.NEAREST)

                    f1, precision, recall, iou = calculate_metrics(true_mask, pred_mask)
                    metrics_data.append((filename, iou, f1, precision, recall))
                    print(f"  -> IoU: {iou:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        # IoU 기준 내림차순 정렬
        metrics_data = sorted(metrics_data, key=lambda x: x[1], reverse=True)

        # 파일에 기록
        for rank, (filename, iou, f1, precision, recall) in enumerate(metrics_data, start=1):
            f.write(f"{rank}\t{filename}\t{iou:.2f}\t{f1:.2f}\t{precision:.2f}\t{recall:.2f}\n")

    print(f"Performance results saved to {output_file}")
    return metrics_data

def main():
    label_dir = r"/home/lsh/share/CD/open-cd/data/10000_total_test/test/1024_label"
    predict_dir = r"/home/lsh/share/CD/open-cd/test/총합/커스텀10000_epoch150_mtp_cross_fold3_total전체10000테스트/1024"
    output_file = r"/home/lsh/share/CD/open-cd/test/총합/커스텀10000_epoch150_mtp_cross_fold3_total전체10000테스트/performance_results_sorted_by_iou.txt"


    print("Starting the image processing...")
    # Process images and calculate metrics
    metrics_data = process_images(label_dir, predict_dir, output_file)

    if metrics_data:
        print("Results have been sorted by IoU and saved.")
    else:
        print("No matching image pairs found or all pairs failed to process.")
    print("Processing completed.")

if __name__ == "__main__":
    main()

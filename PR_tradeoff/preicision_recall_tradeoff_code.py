import argparse
import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def parse_args():
    parser = argparse.ArgumentParser(description='Open-CD test with PR curve generation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='output directory for results')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', help='directory where results will be saved')
    parser.add_argument('--wait-time', type=float, default=2, help='interval of show (s)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

@HOOKS.register_module()
class SaveProbabilityHook(Hook):
    """Hook to save prediction probability maps after testing."""
    def __init__(self, save_dir='prob_maps'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not outputs:
            return
        logits_list = [sample['logits'] for sample in outputs]
        logits = torch.cat(logits_list, dim=0)  # (B,num_classes,H,W)
        prob = torch.softmax(logits, dim=1)
        prob_map = prob[:, 1].detach().cpu().numpy()

        for i, sample in enumerate(outputs):
            img_info = data_batch['data_samples'][i].metainfo
            
            # 리스트 형태의 img_path 처리
            if 'img_path_from' in img_info:
                img_path = img_info['img_path_from']
            elif 'img_path' in img_info:
                img_path = img_info['img_path']
            elif 'filename' in img_info:
                img_path = img_info['filename']
            else:
                raise KeyError("No valid image path key found in metainfo.")

            # img_path가 리스트인 경우 첫 번째 요소 사용
            if isinstance(img_path, list):
                img_path = img_path[0]
            
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(self.save_dir, f'{img_name}_prob.png')
            prob_uint8 = (prob_map[i] * 255).astype('uint8')
            mmcv.imwrite(prob_uint8, save_path)




def patch_model_to_return_logits(runner):
    """Patch the model's test_step to return logits for probability saving."""
    model = runner.model
    orig_test_step = model.test_step

    def forward_test_with_logits(self, data):
        target_size = (256, 256)  # 원하는 출력 사이즈 (H, W)
        
        with torch.no_grad():
            processed = self.data_preprocessor(data)
            x = processed['inputs']
            features = self.extract_feat(x)
            logits = self.decode_head.forward(features)  # (B,num_classes,H,W)
            
            # 모델 출력 (logits)을 256x256으로 리사이즈
            logits_resized = F.interpolate(
                logits, size=target_size, mode='bilinear', align_corners=False
            )  # (B, num_classes, 256, 256)
            pred_sem_seg = logits_resized.argmax(dim=1)  # (B, 256, 256)

            data_samples = []
            for i in range(pred_sem_seg.shape[0]):
                # GT 마스크를 256x256으로 리사이즈
                gt_seg = data['data_samples'][i].gt_sem_seg.data  # PixelData -> Tensor
                gt_seg_resized = F.interpolate(
                    gt_seg.unsqueeze(0).float(),  # (1, H, W)
                    size=target_size,
                    mode='nearest'
                ).squeeze(0).long()  # (256, 256)
                
                data_sample = {
                    'pred_sem_seg': {'data': pred_sem_seg[i].unsqueeze(0)},  # (1, 256, 256)
                    'logits': logits_resized[i].unsqueeze(0),  # (1, num_classes, 256, 256)
                    'gt_sem_seg': {'data': gt_seg_resized}  # 리사이즈된 GT
                }
                data_samples.append(data_sample)
            return data_samples





    from types import MethodType
    model.forward_test_with_logits = MethodType(forward_test_with_logits, model)

    def test_step_with_logits(data):
        return model.forward_test_with_logits(data)

    model.test_step = test_step_with_logits


def analyze_precision_recall(label_dir, prob_dir):
    """Generate precision-recall curve."""
    thresholds = np.linspace(0, 1, 21)
    precision_list = []
    recall_list = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    for thresh in thresholds:
        TP, FP, FN = 0, 0, 0
        for fname in label_files:
            label_path = os.path.join(label_dir, fname)
            prob_path = os.path.join(prob_dir, fname.replace('.png', '_prob.png'))
            if not os.path.exists(prob_path):
                continue
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_binary = (label > 128).astype(np.uint8)

            prob = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            pred_binary = (prob >= thresh).astype(np.uint8)

            TP += np.sum((label_binary == 1) & (pred_binary == 1))
            FP += np.sum((label_binary == 0) & (pred_binary == 1))
            FN += np.sum((label_binary == 1) & (pred_binary == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, marker='o')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig("precision_recall_curve.png")
    plt.close()

    with open("precision_recall_results.txt", 'w') as f:
        f.write("Threshold\tPrecision\tRecall\n")
        for t, p, r in zip(thresholds, precision_list, recall_list):
            f.write(f"{t:.2f}\t{p:.6f}\t{r:.6f}\n")
    print("Precision-Recall analysis done. Results saved.")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # Append custom hook to save probability maps
    if not hasattr(cfg, 'custom_hooks'):
        cfg.custom_hooks = []
    cfg.custom_hooks.append(dict(type='SaveProbabilityHook', save_dir='prob_maps'))

    # Patch model for logits
    runner = Runner.from_cfg(cfg)
    patch_model_to_return_logits(runner)
    runner.test()

    # Analyze Precision-Recall
    analyze_precision_recall(
        label_dir="/home/lsh/share/CD/open-cd/data/5_test/learning/test/label",
        prob_dir="prob_maps"
    )


if __name__ == '__main__':
    main()

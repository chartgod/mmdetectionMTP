import argparse
import logging
import os
import os.path as osp

import torch
import torch.distributed as dist

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from opencd.registry import RUNNERS
from mmengine.registry import init_default_scope


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("--config", default="configs/mtp/rvsa-l-unet-256-mae-mtp_levir.py")
    parser.add_argument("--work-dir", default="rvsa-l-unet-256-mae-mtp_levir_workdir")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()

    # For torch.distributed.launch compatibility
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def distribute_batch_size(total_batch: int, num_gpus: int):
    """총 batch_size를 num_gpus 개로 나누어 GPU별 local batch size 리스트를 반환.
       예) total_batch=32, num_gpus=7 → [5,5,5,5,4,4,4].
    """
    base = total_batch // num_gpus
    remainder = total_batch % num_gpus
    sizes = [base] * num_gpus
    for i in range(remainder):
        sizes[i] += 1
    return sizes


def main():
    init_default_scope('opencd')    
    args = parse_args()

    # 원하는 GPU들 지정 (예: 7장 사용)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

    # DDP 환경 초기화
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

    # Config 로딩
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    # AMP 옵션
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log("AMP training is already enabled in your config.", logger="current", level=logging.WARNING)
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume 옵션
    cfg.resume = args.resume

    # ----------------------------
    # (중요) 여기서 "전체 배치 사이즈" 지정
    # ----------------------------
    desired_global_batch_size = 32  # ← '정확히 32'를 원한다고 가정
    if dist.is_initialized():
        rank = dist.get_rank()        # 현재 프로세스의 rank
        world_size = dist.get_world_size()  # 총 GPU 수

        # 예) 7개 GPU면 -> [5,5,5,5,4,4,4]
        local_bs_list = distribute_batch_size(desired_global_batch_size, world_size)
        local_bs = local_bs_list[rank]

        # 이 rank(GPU)에서만 쓰는 batch size
        cfg.train_dataloader.batch_size = local_bs
        print_log(f"[Rank {rank}] total={desired_global_batch_size} → local batch size={local_bs}", logger="current")
    else:
        # DDP가 아닐 경우(단일 GPU, launcher=none 등), 그냥 32 사용
        cfg.train_dataloader.batch_size = desired_global_batch_size

    # Runner 빌드
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # 학습 시작
    runner.train()


if __name__ == "__main__":
    main()

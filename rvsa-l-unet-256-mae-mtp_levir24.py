############################### default runtime #################################

default_scope = "opencd"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="CDLocalVisBackend")]
visualizer = dict(
    type="CDLocalVisualizer", vis_backends=vis_backends, name="visualizer", alpha=1.0
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

############################### dataset #################################

# dataset settings
dataset_type = "LEVIR_CD_Dataset"
data_root = "/home/lsh/share/open-cd/data/LEVIR-CD256"

crop_size = (256, 256)
train_pipeline = [
    dict(type="MultiImgLoadImageFromFile", imdecode_backend='pillow'),
    dict(type="MultiImgLoadAnnotations"),
    dict(
        type="MultiImgRandomRotFlip", rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)
    ),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgExchangeTime", prob=0.5),
    dict(
        type="MultiImgPhotoMetricDistortion",
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="MultiImgPackSegInputs"),
]

val_pipeline = [
    dict(type="MultiImgLoadImageFromFile", imdecode_backend='pillow'),
    dict(type="MultiImgResize", scale=crop_size, keep_ratio=True),
    dict(type="MultiImgLoadAnnotations"),
    dict(type="MultiImgPackSegInputs"),
]

test_pipeline = [
    dict(type="MultiImgLoadImageFromFile", imdecode_backend='pillow'),
    dict(type="MultiImgLoadAnnotations"),
    dict(type="MultiImgPackSegInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="train/A", img_path_to="train/B", seg_map_path="train/label"
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="val/A", img_path_to="val/B", seg_map_path="val/label"
        ),
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="test/A", img_path_to="test/B", seg_map_path="test/label"
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mFscore", "mIoU"])
test_evaluator = val_evaluator

############################### running schedule #################################


# optimizer
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05),
    constructor="LayerDecayOptimizerConstructor_ViT",
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=0.9,
    ),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type="CosineAnnealingLR",
        T_max=145,
        by_epoch=True,
        begin=1,
        end=5,
    ),
]

# training schedule for 100 epochs
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=150, val_interval=30)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=True, interval=30, save_best="mIoU"
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="CDVisualizationHook", interval=1, img_shape=(256, 256, 3)),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)


############################### running schedule #################################

# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)
model = dict(
    type="SiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="RVSA_MTP",
        img_size=256,
        patch_size=16,
        drop_path_rate=0.3,
        out_indices=[7, 11, 15, 23],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        use_abs_pos_emb=True,
        interval=6,
        frozen_stages=-1,
        pretrained='/home/lsh/share/mmsatellite/pretrained_model/last_vit_l_rvsa_ss_is_rd_pretrn_model_encoder.pth'
    ),
    neck=dict(type="FeatureFusionNeck", policy="abs_diff", out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type="UNetHead",
        num_classes=2,
        ignore_index=255,
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        encoder_channels=[1024, 1024, 1024, 1024],
        decoder_channels=[512, 256, 128, 64],
        n_blocks=4,
        use_batchnorm=True,
        center=False,
        attention_type=None,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)


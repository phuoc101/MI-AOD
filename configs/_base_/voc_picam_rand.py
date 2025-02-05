# Please change the dataset directory to your actual directory
data_root = "/media/phuoc101/imaunicorn/projects/computer_vision/datasets/picam_data/HevoNaTiera/"

# dataset settings
dataset_type = "CustomVOCDataset"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_scaling = (1280, 1280)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=image_scaling, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=image_scaling,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "ImageSets/Main/train_rand.txt",
            img_prefix=data_root,
            pipeline=train_pipeline,
        ),
    ),
    unlabeled=dict(
        type="UnlabeledVOCDataset",
        ann_file=data_root + "ImageSets/Main/unlabeled.txt",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    unlabeled_test=dict(
        type="UnlabeledVOCDataset",
        ann_file=data_root + "ImageSets/Main/unlabeled.txt",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    val=dict(
        type=dataset_type, ann_file=data_root + "ImageSets/Main/val.txt", img_prefix=data_root, pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type, ann_file=data_root + "ImageSets/Main/val.txt", img_prefix=data_root, pipeline=test_pipeline
    ),
)
evaluation = dict(interval=1, metric="mAP")

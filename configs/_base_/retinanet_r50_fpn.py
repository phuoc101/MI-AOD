# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='MIAODRetinaHead',
        C=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # octave_base_scale=4,
            # scales_per_octave=3,
            scales=[0.455, 0.518, 0.697, 1.223, 1.601],
            ratios=[0.495, 0.737, 1.0, 1.357, 2.02],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        FL=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        SmoothL1=dict(type='L1Loss', loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
theta_f_1 = ['bbox_head.f_1_convs.0.conv.weight', 'bbox_head.f_1_convs.0.conv.bias',
             'bbox_head.f_1_convs.1.conv.weight', 'bbox_head.f_1_convs.1.conv.bias',
             'bbox_head.f_1_convs.2.conv.weight', 'bbox_head.f_1_convs.2.conv.bias',
             'bbox_head.f_1_convs.3.conv.weight', 'bbox_head.f_1_convs.3.conv.bias',
             'bbox_head.f_1_retina.weight', 'bbox_head.f_1_retina.bias']
theta_f_2 = ['bbox_head.f_2_convs.0.conv.weight', 'bbox_head.f_2_convs.0.conv.bias',
             'bbox_head.f_2_convs.1.conv.weight', 'bbox_head.f_2_convs.1.conv.bias',
             'bbox_head.f_2_convs.2.conv.weight', 'bbox_head.f_2_convs.2.conv.bias',
             'bbox_head.f_2_convs.3.conv.weight', 'bbox_head.f_2_convs.3.conv.bias',
             'bbox_head.f_2_retina.weight', 'bbox_head.f_2_retina.bias']

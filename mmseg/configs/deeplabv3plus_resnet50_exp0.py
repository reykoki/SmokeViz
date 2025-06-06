norm_cfg = dict(type='SyncBN', requires_grad=True)

hyperparams = dict(
    lr=.001,
    batch_size=8,
    num_workers=4,
    datapointer="/scratch1/RDARCH/rda-ghpcs/Rey.Koki/SmokeViz_code/deep_learning/dataset_pointers/w_null/w_null.pkl"
)

model = dict(
        type='EncoderDecoder',
        pretrained=None,
    backbone=dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True
    ),
    decode_head=dict(
                type='DepthwiseSeparableASPPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=3,
                norm_cfg=norm_cfg,
                align_corners=False,
    ),
    auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=3,
                norm_cfg=norm_cfg,
                align_corners=False,
    ),
)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
        pretrained=None,
        type='EncoderDecoder',
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
                type='PSPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                pool_scales=(1, 2, 3, 6),
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
    )
)


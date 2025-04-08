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
                type='mmpretrain.ConvNeXt',
                arch='base',
                out_indices=[0, 1, 2, 3],
                drop_path_rate=0.4,
                layer_scale_init_value=1.0,
                gap_before_final_norm=False,
    ),
    decode_head=dict(
                type='UPerHead',
                in_channels=[128, 256, 512, 1024],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=3,
                norm_cfg=norm_cfg,
                align_corners=False,
    ),
    auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
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

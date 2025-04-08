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
                type='ResNet',
                depth=18,
                num_stages=4,
                in_channels=3,
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                out_indices=(0,1,2,3),
                norm_cfg=norm_cfg,
    ),
    decode_head=dict(
                type='PSPHead',
                in_channels=128,
                in_index=1,
                channels=256,
                num_classes=3,
                up=True,
                norm_cfg=norm_cfg
    )
)

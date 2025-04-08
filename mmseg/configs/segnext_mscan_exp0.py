ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

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
        type='MSCAN',
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),

    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
)


_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 32
fixed_beta = 49.17
age_weights = {
    "1": 0.0029,
    "2": 0.0038,
    "3": 0.0024,
    "4": 0.0019,
    "5": 0.0021,
    "6": 0.0022,
    "7": 0.0022,
    "8": 0.0022,
    "9": 0.0025,
    "10": 0.0029,
    "11": 0.003,
    "12": 0.0029,
    "13": 0.0034,
    "14": 0.0041,
    "15": 0.0041,
    "16": 0.0032,
    "17": 0.0036,
    "18": 0.0045,
    "19": 0.0032,
    "20": 0.0026,
    "21": 0.0022,
    "22": 0.0022,
    "23": 0.0025,
    "24": 0.0025,
    "25": 0.0026,
    "26": 0.0032,
    "27": 0.0029,
    "28": 0.0029,
    "29": 0.0024,
    "30": 0.0023,
    "31": 0.0025,
    "32": 0.0025,
    "33": 0.0026,
    "34": 0.0027,
    "35": 0.0028,
    "36": 0.0027,
    "37": 0.0032,
    "38": 0.0025,
    "39": 0.0023,
    "40": 0.0023,
    "41": 0.0024,
    "42": 0.0017,
    "43": 0.0019,
    "44": 0.0025,
    "45": 0.0022,
    "46": 0.0025,
    "47": 0.0023,
    "48": 0.0019,
    "49": 0.0027,
    "50": 0.0026,
    "51": 0.0032,
    "52": 0.0022,
    "53": 0.0028,
    "54": 0.0032,
    "55": 0.0034,
    "56": 0.0041,
    "57": 0.003,
    "58": 0.0028,
    "59": 0.0051,
    "60": 0.0022,
    "61": 0.0041,
    "62": 0.0026,
    "63": 0.0028,
    "64": 0.0041,
    "65": 0.0051,
    "67": 0.0019,
    "68": 0.0051,
    "69": 0.0058,
    "70": 0.0036,
    "71": 0.0071,
    "72": 0.0051,
    "73": 0.0071,
    "74": 0.0058,
    "75": 0.0051,
    "76": 0.0101,
    "78": 0.0071,
    "80": 0.0071,
    "81": 0.0071,
    "84": 0.0051,
    "85": 0.0051,
    "86": 0.0071,
    "87": 0.0045,
    "88": 0.0045,
    "89": 0.0045,
    "90": 0.0058,
    "91": 0.0051,
    "92": 0.0058,
    "93": 0.0038,
    "94": 0.0045,
    "95": 0.0051,
    "96": 0.0058,
    "97": 0.0058,
    "98": 0.0101,
    "99": 0.0058,
    "100": 0.0045,
    "101": 0.0051,
    "102": 0.0051,
    "104": 0.0051,
    "106": 0.0051,
    "108": 0.0058,
    "109": 0.0041,
    "112": 0.0058,
    "115": 0.0101,
    "116": 0.0071,
    "117": 0.0058,
    "119": 0.0071,
    "120": 0.0101,
    "123": 0.0071,
    "124": 0.0051,
    "126": 0.0071,
    "127": 0.0101,
    "128": 0.0101,
    "130": 0.0058,
    "133": 0.0101,
    "138": 0.0071,
    "139": 0.0051,
    "140": 0.0045,
    "141": 0.0101,
    "142": 0.0041,
    "143": 0.0101,
    "145": 0.0071,
    "146": 0.0058,
    "147": 0.0071,
    "149": 0.0101,
    "150": 0.0101,
    "152": 0.0101,
    "153": 0.0058,
    "155": 0.0101,
    "156": 0.0038,
    "157": 0.0045,
    "160": 0.0071,
    "162": 0.0101,
    "163": 0.0071,
    "168": 0.0071,
    "169": 0.0101,
    "171": 0.0071,
    "172": 0.0101,
    "173": 0.0101,
    "174": 0.0071,
    "175": 0.0071,
    "176": 0.0071,
    "179": 0.0101,
    "180": 0.0071,
    "182": 0.0101,
    "186": 0.0101,
    "187": 0.0101,
    "188": 0.0101,
    "199": 0.0101,
    "201": 0.0101,
    "202": 0.0071,
    "203": 0.0101,
    "206": 0.0101,
    "208": 0.0101,
    "209": 0.0058,
    "210": 0.0101,
    "211": 0.0058,
    "212": 0.0071,
    "214": 0.0071,
    "215": 0.0058,
    "216": 0.0101,
    "219": 0.0101,
    "220": 0.0071,
    "221": 0.0101,
    "222": 0.0101,
    "223": 0.0101,
    "225": 0.0101,
    "255": 0.0101,
    "258": 0.0101,
    "272": 0.0101,
    "289": 0.0101,
    "291": 0.0101,
    "313": 0.0101,
    "334": 0.0101,
    "348": 0.0101
}

test_output_dir = '/workspace/work_dirs/load_ff3d_forage/test_forage2/round2/vote9'
test_or_val = 'test'
model = dict(
    type='ForestFormerDownstream_Age', 
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,
    num_channels=num_channels,
    voxel_size=0.05,  #default 0.05
    min_spatial_shape=128,
    fixed_beta = fixed_beta,
    age_weights = age_weights,
    test_output_dir = test_output_dir,
    test_or_val = test_or_val,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    train_cfg=dict(),
    test_cfg=dict()
    )

# dataset settings
dataset_type = 'forageDataset'
data_root_forage = 'data/forage/'
data_prefix = dict(
    pts='points',
    age_label='age_label')
#class_names_forage = (
#    'tree')
#metainfo_forage = dict(
#    classes=class_names_forage,
#    ignore_index=num_semantic_classes)

train_pipeline = [
    dict(
        type='LoadPointsAndAgeLabelFromFile',
        coord_type='DEPTH',
        shift_height=False,
        shift_height_xyz=True,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.01, 0.01, 0.01],
        clip_range=[-0.05, 0.05]),
    dict(type='RandomGridSample', grid_size_range=(0.01, 0.1)),
    dict(
        type='PointSample_',
        num_points=20000),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'age_label'
        ])
]
val_pipeline = [
    dict(
        type='LoadPointsAndAgeLabelFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='PointSample_',
        num_points=20000),
    dict(type='Pack3DDetInputs_', keys=['points', 'age_label'])
]
test_pipeline = [
    dict(
        type='LoadPointsAndAgeLabelFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='PointSample_',
        num_points=20000),
    dict(type='Pack3DDetInputs_', keys=['points', 'age_label'])
]

# run settings
train_dataloader = dict(
    batch_size=64,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forage,
        ann_file='forage_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline))
val_dataloader = dict(
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forage,
        ann_file='forage_oneformer3d_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forage,
        ann_file='forage_oneformer3d_infos_test.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True))

#class_names += ['unlabeled']

sem_mapping = [
    0, 1, 2]
inst_mapping = sem_mapping[1:]
val_evaluator = dict(
    type='ForAgeRegMetric'
   )

test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05), 
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=58984, power=0.9, by_epoch=False) #end=num_samples/batch_size*max_epochs

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
#default_hooks = dict(
#    checkpoint=dict(interval=1, max_keep_ckpts=3))
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
        save_best='RMSE',
        rule='less'),
        logger=dict(type='LoggerHook', interval=30),
        visualization=dict(type='Det3DVisualizationHook', draw=False))
#default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
#log_config = dict(
#    interval=50,
#    hooks=[
#        dict(type='TextLoggerHook'),
#        dict(type='TensorboardLoggerHook')
#    ])

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

load_from = '/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth'
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5000,
    val_interval=30)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = True

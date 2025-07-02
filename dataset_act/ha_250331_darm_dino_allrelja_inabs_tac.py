"""params settings."""
# train params
import torch
import socket
import copy
import os
import numpy as np

total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1000)
if total_memory < 16:
    batch_size = 2
    workers = 1
elif 80<total_memory<90:
    batch_size = 81
    workers = 12
elif 90<total_memory<100:
    batch_size = 85
    workers = 18

pin_memory = False
num_gpu = torch.cuda.device_count()  # NOTE: CUDA_VISIBLE_DEVICES

max_iters = 200000
save_interval = 2000
val_interval = max_iters

# others params
file_client_args = dict(backend="disk")
opencv_num_threads = 0
mp_start_method = "fork"

train_time = "2503"
train_task = "dh_photography"
train_task_version = "v4"
control_mode = "all_rel"
base_dir = f"{train_time}/{train_task}_{train_task_version}"
work_dir_base = f"work_dirs/data_server/logs/hesai/tel/{base_dir}/"
dev_hostname =  os.getenv("SLURM_JOB_ID", socket.gethostname())
if dev_hostname == socket.gethostname():
    if os.path.exists("/mnt/netdata/"):
        netdisk = "/mnt/netdata/Team/Robot/Data/train/"
    elif os.path.exists("/mnt/net-cloud4/"):
        netdisk = "/mnt/net-cloud4/Team/Robot/train/"
    else:
        raise NotImplementedError
else:
    netdisk = "/work/share/acrx9u6hcm/" # NOTE: netdisk on scnet
    work_dir_base += "@scnet/"

"""
data settings
"""
### 遥操作搭积木数据
TMODE1 = [
    # f"{netdisk}{train_task}/{train_task_version}/0313_TMODE1/season_arm01_2025_03_13_19_30_37_train",
    # f"{netdisk}{train_task}/{train_task_version}/0314_TMODE1/season_arm01_2025_03_14_14_56_07_train",
    # f"{netdisk}{train_task}/{train_task_version}/0315_TMODE1/season_arm01_2025_03_15_10_39_31_train",
    # f"{netdisk}{train_task}/{train_task_version}/0315_TMODE1/season_arm01_2025_03_17_10_50_38_train",
    # f"{netdisk}{train_task}/{train_task_version}/0315_TMODE1/season_arm01_2025_03_18_09_58_19_train",
    # f"{netdisk}{train_task}/{train_task_version}/0315_TMODE1/season_arm01_2025_03_19_10_04_06_train",
    f"{netdisk}{train_task}/{train_task_version}/0328_TMODE1/season_UHR02_2025_03_27_16_11_43_train",
    f"{netdisk}{train_task}/{train_task_version}/0328_TMODE1/season_UHR02_2025_03_28_16_23_04_train",
    f"{netdisk}{train_task}/{train_task_version}/0328_TMODE1/season_UHR02_2025_03_29_10_51_56_train",
    f"{netdisk}{train_task}/{train_task_version}/0328_TMODE1/season_UHR02_2025_03_31_10_48_25_train",
    f"{netdisk}{train_task}/{train_task_version}/0328_TMODE1/season_UHR02_2025_03_31_17_11_31_train", 
]

### 数据的定义
DATA_TYPE="0.2.2"   # NOTE： action默认60Hz, observe默认30Hz, 模型前向过程默认10Hz
personal_dataroot="/mnt/netdata/Team/Robot/lhy/data_server"
norm_stats_cache=f"{personal_dataroot}/data/hesai/tel/{base_dir}_{control_mode}_tac.pkl"
inference_dt = 0.1
collect_dt = 0.033

dataset_config = dict(
    dataset_type="HaPipelineV2DatasetD020",
    seed=0,
    data_root=TMODE1,
    # seq_list
    train_skip=None,
    test_skip=None,
    train_seq_list=None,
    test_seq_list=None,
    # obs information
    train_ratio=0.95,
    process_range_desc=[
        "/anno/start_move/step_table",
        # "/anno/dagger/step_table"
    ],  # NOTE: filter start point
    sample_rate=int(inference_dt/collect_dt),
    tac_sample_rate=1,  # NOTE: 高频触觉
    collect_dt=collect_dt,
)

## pipeline的输入数据局定义
camera_preprcess_info=[
    # dict(
    #     name='/observe/vision/chest/rgbd/rgb',
    #     # crop=[27,103,552,660],
    #     # transpose=True,
    #     img_size=[180, 320],
    # ),
    dict(
        name='/observe/vision/head/stereo/lefteye/rgb',
        # crop=[27,103,552,660],
        # transpose=True,
        img_size=[180, 320],
    ),
    dict(
        name='/observe/vision/head/stereo/righteye/rgb',
        # crop=[27,103,552,660],
        # transpose=True,
        img_size=[180, 320],
    ),
    dict(
        name='/observe/vision/right_wrist/fisheye/rgb', # 原图倒转
        flip="xy",
        crop=[27, 103, 552, 660],
        # transpose=True,
        img_size=[256, 280],
    ),
    dict(
        name='/observe/vision/left_wrist/fisheye/rgb', # 原图倒转
        flip="xy",
        crop=[27, 103, 552, 660],
        # transpose=True,
        img_size=[256, 280],
    ),
]
camera_names = list(c["name"] for c in camera_preprcess_info)

h5_lowdim_input = [
    "/state/right_arm/joint_angle",
    "/state/right_hand/joint_angle",
    "/state/left_arm/joint_angle",
    "/state/left_hand/joint_angle",
    "/state/neck/joint_angle",
]
h5_action_input = [
    "/action/right_arm/joint_angle",
    "/action/right_hand/joint_angle",
    "/action/left_arm/joint_angle",
    "/action/left_hand/joint_angle",
    "/action/neck/joint_angle",
]

#### 模型的数据数据定义
lowdim_to_model = [
    "/state/right_arm/joint_angle",
    "/state/right_hand/joint_angle",
    "/state/left_arm/joint_angle",
    "/state/left_hand/joint_angle",
    "/state/neck/joint_angle",
]   # qpos
action_to_model = [
    "/action/right_arm/joint_angle/rel",
    "/action/right_hand/joint_angle/rel",
    "/action/left_arm/joint_angle/rel",
    "/action/left_hand/joint_angle/rel",
    "/action/neck/joint_angle/rel",
]

hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
neck_mask = [1, 1]
arm_mask = [1 for _ in range(7)]
model_mask_list = [arm_mask, hand_mask, arm_mask, hand_mask, neck_mask]
model_mask = [i for mask in model_mask_list for i in mask]
model_state_dim_list = [7, 21, 7, 21, 2]
model_action_dim_list = [7, 21, 7, 21, 2]
model_state_dim = sum(model_state_dim_list)
model_action_dim = sum(model_action_dim_list)

### tactile
h5_tactile_input = [
    "/observe/tactile/right_thumb/force",
    "/observe/tactile/right_thumb/torque",
    "/observe/tactile/right_index/force",
    "/observe/tactile/right_index/torque",
    "/observe/tactile/right_middle/force",
    "/observe/tactile/right_middle/torque",
    "/observe/tactile/right_ring/force",
    "/observe/tactile/right_ring/torque",
    "/observe/tactile/right_little/force",
    "/observe/tactile/right_little/torque",
    "/observe/tactile/left_thumb/force",
    "/observe/tactile/left_thumb/torque",
    "/observe/tactile/left_index/force",
    "/observe/tactile/left_index/torque",
    "/observe/tactile/left_middle/force",
    "/observe/tactile/left_middle/torque",
    "/observe/tactile/left_ring/force",
    "/observe/tactile/left_ring/torque",
    "/observe/tactile/left_little/force",
    "/observe/tactile/left_little/torque",
]
tacforce_name = "/observe/tactile/total_force"
model_tac_state_dim_list = [3 for _ in range(len(h5_tactile_input))]
model_tac_state_dim = sum(model_tac_state_dim_list)
lowdim_to_normalize = lowdim_to_model + action_to_model + [tacforce_name]

history_cnt = 6    # NOTE: history帧数
chunk_size = 100
tac_history_cnt = int(history_cnt*dataset_config["sample_rate"]/float(dataset_config["tac_sample_rate"]))  # 3*6/1=18

"""
save dir
"""
work_dir = os.path.join(
    work_dir_base, 
    f"250331_act_{control_mode}_ja_mse_{len(camera_names)}dino_chunk{chunk_size}_qpos{history_cnt}_tac{tac_history_cnt}"
)

load_from = None

'''
input and pipeline
'''
lowdim_load_pipeline = [
    # state loader
    *[dict(
        target="robotsdl.datasets.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[0, chunk_size],
        step=1,
        padding_value="closest",
        hf_conf=dict(
            index_key="aligned_index",
            use_index_value=False,  # NOTE: 高频action输出
        ),
        dtype='float32',
    ) for name in h5_action_input],
    *[dict(
        target="robotsdl.datasets.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[-(history_cnt - 1) * dataset_config["sample_rate"], 1],
        step=dataset_config["sample_rate"],
        padding_value="closest",
        hf_conf=dict(
            index_key="aligned_index",
            use_index_value=True,   # NOTE: 低频history输入
        ),
        dtype='float32',
    ) for name in h5_lowdim_input],
    *[dict(
        target="robotsdl.datasets.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[-(tac_history_cnt - 1) * dataset_config["tac_sample_rate"], 1],
        step=dataset_config["tac_sample_rate"],  # NOTE: 高频30Hz触觉信号输入
        padding_value="closest",
        hf_conf=None,
        dtype='float32',
    ) for name in h5_tactile_input],
]

img_load_pipeline = [
    *[dict(
        target="robotsdl.datasets.pipelines_v2.slice_loader.ImageLoader",
        key=name,
        range=[0, 1],
        step=dataset_config["sample_rate"],
        padding_value="closest",
    ) for name in camera_names],
]


lowdim_process_pipeline = [
    *[dict(
        target="robotsdl.datasets.pipelines_v2.transform.RelativeGenerate",
        key=name,
        ref_key=h5_lowdim_input[i],
        meta={"type": "rel_abs"},
    ) for i, name in enumerate(h5_action_input)],
    dict(
        target="robotsdl.datasets.pipelines_v2.transform.ConcatData",
        keys=h5_tactile_input,
        output_key=tacforce_name,  # [L, C]
        dim=-1,
    ),
    dict(
        target="robotsdl.datasets.pipelines_v2.transform.RelativeGenerate",
        key=tacforce_name,
        ref_key=tacforce_name,
        ref_index=0,
        meta={"type": "rel_abs"},
    ),
    dict(
        target="robotsdl.datasets.pipelines_v2.transform.ConcatData",
        keys=[tacforce_name, f"{tacforce_name}/rel"],
        output_key=tacforce_name,  # [L, 2C], abs+rel
        dim=-1,
    ),
]

img_process_pipeline = [
    ### image processor
    *[dict(
        target="robotsdl.datasets.pipelines_v2.transform.ImageProcess",
        key=cam_info["name"],
        norm=True,
        aug=True,
        aug_geo=True,
        aug_color=True,
        camera_preprcess_info=cam_info,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        aug_conf=dict(
            crop_ratio=0.98,
            rotate_range=[-0.3, 0.3],
            brightness_delta=20,
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2)
        ),
        debug=False,
    ) for cam_info in camera_preprcess_info],
]
load_pipeline = lowdim_load_pipeline + img_load_pipeline
process_pipeline = lowdim_process_pipeline + img_process_pipeline
stat_pipeline = lowdim_load_pipeline + lowdim_process_pipeline
train_pipeline = load_pipeline + process_pipeline

test_pipeline = []
for pipe in copy.deepcopy(train_pipeline):
    if pipe["target"] == "robotsdl.datasets.pipelines_v2.transform.ImageProcess":
        pipe["aug"] = False
    test_pipeline.append(pipe)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(
        data_type=DATA_TYPE,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=False,
        norm_stats_cache=norm_stats_cache,
        pipeline=train_pipeline,
        stat_pipeline=stat_pipeline,
        stat_sample_step=10,
        stat_worker=16,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("train_skip", None),
        seq_list=dataset_config.get("train_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
    val=dict(
        data_type=DATA_TYPE,
        limit_num=batch_size * 20,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=True,
        norm_stats_cache=norm_stats_cache,
        pipeline=test_pipeline,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("test_skip", None),
        seq_list=dataset_config.get("test_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
    test=dict(
        data_type=DATA_TYPE,
        limit_num=batch_size * 20,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=True,
        norm_stats_cache=norm_stats_cache,
        pipeline=test_pipeline,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("test_skip", None),
        seq_list=dataset_config.get("test_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
)

"""
model settings
"""

norm_method = "gaussian"
# hesai_finger_index = ["pinky", "ring", "middle", "index", "thumb", "arm_pinky"]
urdf_root_path = "/mnt/netdata/Team/Robot/lhy/data_server/assert/urdf/rm_lb_hand_fk"
model = dict(
    type="HaVAEPolicyV2",
    kl_weight=10,
    preprocess={
        **{name + "_normalize": dict(
            target="robotsdl.models.preprocess.process_v2.Normalize",
            method=norm_method,
            input_name=[name, "norm_stats"],
        ) for name in lowdim_to_normalize},
        "mask_right_hand_state": dict(
            target="robotsdl.models.preprocess.process_v2.MaskData",
            mask=hand_mask,
            input_name=["/state/right_hand/joint_angle/norm"]),
        "mask_left_hand_state": dict(
            target="robotsdl.models.preprocess.process_v2.MaskData",
            mask=hand_mask,
            input_name=["/state/left_hand/joint_angle/norm"]),
        "concat_qpos": dict(
            target="robotsdl.models.preprocess.process_v2.ConcatData",
            input_name=[f"{name}/norm" for name in lowdim_to_model],
            output_key="qpos/norm",
        ),
        "concat_action": dict(
            target="robotsdl.models.preprocess.process_v2.ConcatData",
            dim=-1,
            input_name=[f"{name}/norm" for name in action_to_model],
            output_key="actions/norm/gt",
        ),
        "rename_action_padding": dict(
            target="robotsdl.models.preprocess.process_v2.RenameKey",
            dim=-1,
            input_name=[
                f"{h5_action_input[-1]}/is_padding",
            ],
            output_key="is_pad",
        ), 
        "wrap_imgs": dict(
            target="robotsdl.models.preprocess.process_v2.WrapData",
            input_name=camera_names,
            output_key="imgs",
        ),
    },
    vae_encoder=dict(
        type="HaVaeEncoderV2",
        inference_key="qpos/norm",
        random_sample=False,
        latent_dim=32,
        pos_embed=dict(
            type="learned",
            start_num=0,
        ),
        encoder=dict(
            type="HaDETRTransformerEncoder",
            batch_first=True,
            d_model=512,
            nhead=8,
            dim_feedforward=3200,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            num_layers=4,
        ),
        token_encoder=dict(
            action_backbones=dict(
                input_name=["actions/norm/gt", "is_pad"],
                type="HaActionCLSProj",
                action_token_num=chunk_size,
                action_dim=model_action_dim,
                hidden_dim=256,
                output_dim=512,
                with_cls_embed=True,
                with_pos_embed=True,
                pos_embed=dict(
                    type="learned"
                ),
            ),
            state_backbones=dict(
                input_name=["qpos/norm"],
                type="HaStateProj",
                state_token_num=history_cnt, 
                state_dim=model_state_dim, 
                hidden_dim=256, 
                output_dim=512, 
                with_pos_embed=True,
                pos_embed=dict(
                    type="learned"
                ),
            ),
        )
    ),
    decoder=dict(
        type="DETRMultiModalV2",
        chunk_size=chunk_size,
        n_frames_info=history_cnt,
        output_idx=-1,  # TODO, -1
        wanted_output=None,
        encoder=dict(
            type="HaDETRTransformerMultiModal",
            d_model=512,
            n_head=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=3200,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
        ),
        token_encoder=dict(
            state_backbones=dict(
                input_name=["qpos/norm"],
                type="HaStateProj",
                state_token_num=history_cnt, 
                state_dim=model_state_dim,
                hidden_dim=256, 
                output_dim=512, 
                with_pos_embed=True,
                pos_embed=dict(
                    type="learned"
                ),
            ),
            tactile_backbones=dict(
                input_name=[tacforce_name],
                target="robotsdl.models.backbones.backbones_1d.wavenet_proj.WaveNetprojV1",
                state_token_num=3, 
                wavenet_conf=dict(
                    layer_size=4,   # 1+2+4+8=15 
                    stack_size=1, 
                    in_channels=2*model_tac_state_dim, 
                    res_channels=128,
                    out_channels=512,
                ),
                with_pos_embed=True,
                pos_embed=dict(
                    type="learned"
                ),
            ),
            image_backbones=dict(
                input_name=["imgs"],
                target="robotsdl.models.backbones.backbones_2d.timm.HaTimmWrapper",
                camera_names=camera_names,
                # total_token_num=1346,
                total_token_num=len(camera_names),
                output_dim=512,
                feat_agg_method="slice",
                camera_feat_agg_conf=[{
                    "camera_token_num": 1, # cls token only for vit
                } for _ in range(len(camera_names))],
                # feat_agg_method="all",
                # camera_feat_agg_conf=None,
                wrapper_type=dict(
                    type="seperate",
                ),
                encoder_conf=[
                    dict(
                        # vit_large_patch14_dinov2.lvd142m 1.22G out:1024
                        # vit_base_patch14_dinov2.lvd142m 346M out:768
                        # vit_small_patch14_dinov2.lvd142m 13.33M out:384
                        model_name="vit_base_patch14_dinov2.lvd142m",  
                        pretrained=True,
                        img_size=cam_info["img_size"],
                        global_pool=''
                    ) for cam_info in camera_preprcess_info
                ],
                encoder_output_dim=768,
                with_pos_embed=True,
                pos_embed=dict(
                    type="learned",
                    start_num=2,
                ),
            )
        )
    ),
    head={
        "action_and_loss" : dict(
            type="HaPredL1Head",
            input_latent_dim=512,
            action_dim=model_action_dim,
            mse_loss_weight=model_mask,
            input_name=["latent_feat", "actions/norm/gt", "is_pad"],
        ),
        # split pred
        "split_pred" : dict(
            target="robotsdl.models.heads.base_v2_head.SplitLowDimHead",
            input_name=["actions/norm/pred"],
            split_keys=[f"{name}/norm/pred" for name in action_to_model],
            split_ranges=[[0, 7], [7, 28], [28, 35], [35, 56], [56, 58]],
        ),

        # denormalize
        **{
            f"denormalize_{name}": dict(
                target="robotsdl.models.preprocess.process_v2.Denormalize",
                method=norm_method,
                input_name=[f"{name}/norm/pred", "norm_stats"],
            ) for name in action_to_model
        },
        # remove rel
        **{
            f"parse_rel_{name.split('/')[2]}": dict(
                target="robotsdl.models.heads.base_v2_head.ParseRelativeHead",
                input_name=[f"/action/{name.split('/')[2]}/joint_angle/rel/pred",
                            f"/state/{name.split('/')[2]}/joint_angle"],
                meta={"type": "rel_abs"},
                ref_index=-1,
            ) for name in h5_action_input
        },

        **{f"joint_eef_abs_{part}": dict(
                target="robotsdl.models.heads.ha_ja2eef_head.HaJa2EefHeadV3",
                robot_urdf_path=f"{urdf_root_path}/realman.urdf",
                target_links=["Link7"],
                proto_joint_names=[f"joint{i}" for i in range(1, 8)],
                urdf_base_on_eff=None,
                with_rot_loss=True, 
                rot_loss_weight=0.4,
                rot_dim=6,
                prefix=f"{part}",
                input_name=[f"/action/{part}/joint_angle/pred", 
                            f"/action/{part}/joint_angle",
                            "is_pad"],  # hand relative
            ) for part in ["left_arm", "right_arm"]
        },
        
        # "merge_pred": dict(
        #     target="robotsdl.models.preprocess.process_v2.ConcatData",
        #     dim=-1,
        #     input_name=[f"{n.replace('/rel','')}/pred" for n in action_to_model[:-1]],  # 除去neck
        #     output_key="/action/merge_action/pred"
        # ),
        # "merge_gt": dict(
        #     target="robotsdl.models.preprocess.process_v2.ConcatData",
        #     dim=-1,
        #     input_name=[n.replace('/rel','') for n in action_to_model[:-1]],
        #     output_key="/action/merge_action"
        # ),
        # "joint_eef_abs_hand": dict(
        #     target="robotsdl.models.heads.ha_ja2eef_head.HaJa2EefHeadV3",
        #     robot_urdf_path=f"{urdf_root_path}/rm_luban_hand.urdf",
        #     target_links=[f"Thumb5", f"Index3"],
        #     proto_joint_names=[f"joint{i}" for i in range(1, 8)] + ['P4', 'P3', 'P1', 'P2', 'R4', 'R3', 'R1', 'R2', 'M4', 'M3', 'M1', 'M2', 'I4', 'I3', 'I1', 'I2', 'T5', 'T3', 'T4', 'T2', 'T1'],
        #     urdf_base_on_eff=None,
        #     with_rot_loss=True, 
        #     rot_loss_weight=0.4,
        #     rot_dim=6,
        #     input_name=["/action/merge_action/pred", 
        #                 "/action/merge_action",
        #                 "is_pad"],  # hand relative
        # ),

        ## sumarry final loss
        "summary_loss": dict(
            target="robotsdl.models.heads.base_v2_head.SummaryLossHead",
            weigths=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            input_name=[
                "actions/norm/l1", 
                "kl_loss",
                # "Thumb5/eef_pos_l2",
                # "Thumb5/eef_rot_l1",
                # "Index3/eef_pos_l2",
                # "Index3/eef_rot_l1",
                "right_arm/Link7/eef_pos_l2",
                "right_arm/Link7/eef_rot_l1",
                "left_arm/Link7/eef_pos_l2",
                "left_arm/Link7/eef_rot_l1",
            ],
        ),

        # loss just for verbose
        **{
            f"verbose_{name.split('/')[2]}_ja_l1": dict(   # ja-> joint angle
                target="robotsdl.models.heads.base_v2_head.L1LossHead",
                input_name=[
                    f"/action/{name.split('/')[2]}/joint_angle/rel/pred",
                    f"/action/{name.split('/')[2]}/joint_angle/rel",
                    "is_pad"
                ],
                weights=model_mask_list[i], # TODO: mask
            ) for i, name in enumerate(h5_action_input)
        },
        # make pred as input name
        **{f"unwrapp_predict_{name}": dict(
            target="robotsdl.models.heads.base_v2_head.UnwrapActionPred",
            input_name=[f"{name}/pred"],    # output remove pred
        ) for name in h5_action_input},
    }
)


"""
schedules settings
"""
find_unused_parameters = True
# fp16=dict(loss_scale='dynamic')

runner = dict(type="RobotsIterBasedRunner", max_iters=max_iters)  # MSS MTL

evaluation = dict(
    interval=val_interval,
    by_epoch=False,
    return_loss=True,
)

optimizer = dict(
    type="AdamW",
    lr=1.0e-4,
    # betas=(0.9, 0.99),
    weight_decay=1e-4,
    custom_keys=dict(
        backbone=dict(
            lr=1.0e-5,
        ),
    ),
)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
optimizer_config = dict()

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-1,
)

"""
basic settings
"""
dist_params = dict(backend="nccl")
log_level = "INFO"
resume_from = None


"""
runtime settings
"""
checkpoint_config = dict(interval=save_interval, max_keep_ckpts=2)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False, average=False),
        # dict(type="TensorboardLoggerHook", by_epoch=False, average=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="robot-camera-v4-3.0",
                name=f"{work_dir.split('/')[-1]}({dev_hostname})",
                entity="leihanyue-sjtu",
                resume=False,
                dir=work_dir,
                id=None,
            ),
            by_epoch=False,
            average=False
        ),
        dict(type="InfoHook", by_epoch=False, interval=100),
    ],
)


"""
sim
"""
env_pipeline = [
    # state loader
    *copy.deepcopy(process_pipeline),

    # add batch dim
    dict(
        target="robotsdl.datasets.pipelines_v2.transform.EnvAddBatchDim",
        keys=set(camera_names + lowdim_to_model + h5_lowdim_input + [tacforce_name]), # for parse reference
    ),
]
for pi, pipe in enumerate(env_pipeline):
    if pipe["target"] == "robotsdl.datasets.pipelines_v2.transform.ImageProcess":
        pipe["aug"] = False

dt = dataset_config["collect_dt"] * dataset_config["sample_rate"]
env = dict(
    type="Ros2Env",
    num_queries=chunk_size,
    real_robot=False,
    onscreen_render=False,
    num_rollouts=100,
    BOX_POSE=[None],
    onscreen_cam="angle",
    # state_dim=model["decoder"]["state_dim"],
    preset_image_input=None,
    # policy_class=info_dict["policy_class"],
    norm_stats_cache=norm_stats_cache,
    load_pipeline=load_pipeline,
    # action_dim=info_dict["action_dim"],
    robot_type="v3",    # support v2-remote system
    inference_type="v2",
    obs_timestamp_key=None,

    inference_dt=inference_dt,   # NOTE: 模型action发送频率
    action_output=h5_action_input,  # NOTE: model输出action值
    temporal_agg=[
        dict(
            action_name="/action/right_arm/joint_angle",
            agg_flag=True,
            train_dt=0.01666,
            offset=0,
            output_chunk=40,
            with_raw=True,
            with_fk=dict(
                robot_urdf_path=f"{urdf_root_path}/realman.urdf",
                target_links=["Link7"],
                proto_joint_names=[f"joint{i}" for i in range(1, 8)],
            )
        ),
        dict(
            action_name="/action/right_hand/joint_angle",
            agg_flag=True,
            train_dt=0.01666,
            offset=0,
            output_chunk=40,
        ),
        dict(
            action_name="/action/left_arm/joint_angle",
            agg_flag=True,
            train_dt=0.01666,
            offset=0,
            output_chunk=40,
            # with_raw=True,
            # with_fk=dict(
            #     robot_urdf_path=f"{urdf_root_path}/realman.urdf",
            #     target_links=["Link7"],
            #     proto_joint_names=[f"joint{i}" for i in range(1, 8)],
            # )
        ),
        dict(
            action_name="/action/left_hand/joint_angle",
            agg_flag=True,
            train_dt=0.01666,
            offset=0,
            output_chunk=40,
        ),
        dict(
            action_name="/action/neck/joint_angle",
            agg_flag=True,
            train_dt=0.01666,
            offset=0,
            output_chunk=40,
        )
    ], # NOTE: 和输出action对应，可配置是否进行agg
    debug=False,
    custom_obs=dict(
        **{
            # 先load keep_length 个，然后选择其中的某些位置的，可以用于降频等，升频暂时只能用长度1+传来的高频list
            f"{observe}":dict(
                keep_length=1,
                # add_his_dim=False,  # default: True
                # choosen=[0],
            ) for observe in h5_tactile_input
        }
    )
)
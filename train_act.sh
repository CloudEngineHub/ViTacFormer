python3 imitate_episodes.py \
--task_name dh_photography \
--ckpt_dir ckpt_dir/dh_phtography \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-4 \
--seed 0 \
--use_tactile \
# --resume_path /mnt/home/lheng/heng/RoboVerse/roboverse_learn/algorithms/act/ckpt_dir/dh_hamburger/20250408_232544_tactile/policy_epoch_0_loss_1.5090279579162598_l1_0.11074136942625046.ckpt

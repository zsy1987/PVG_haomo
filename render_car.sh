CUDA_VISIBLE_DEVICES=3 python evaluate.py \
--config configs/haomo_reconstruction.yaml \
source_path=/mnt/ve_share3/share/air/HC04 \
model_path=./eval_output/haomo_reconstruction/ \
start_frame=0 end_frame=98
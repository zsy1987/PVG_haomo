CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/haomo_reconstruction.yaml \
source_path=../HC04 \
model_path=../eval_output/haomo_reconstruction/ \
start_frame=0 end_frame=98
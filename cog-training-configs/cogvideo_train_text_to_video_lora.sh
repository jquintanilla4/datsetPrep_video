export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export OMP_NUM_THREADS=16

GPU_IDS="0"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("4000")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/mnt/d/cogvideox-factory/datasets/cinecam"

CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
MODEL_PATH="THUDM/CogVideoX1.5-5B"

# Set ` --load_tensors ` to load tensors from disk instead of recomputing the encoder process.
# Launch experiments with different hyperparameters

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./models/cinecam/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_text_to_video_lora.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 16 24 32 40 64 72 \
          --dataloader_num_workers 8 \
          --seed 42 \
          --rank 256 \
          --lora_alpha 256 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 72 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 1000 \
          --resume_from_checkpoint latest \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 100 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --enable_model_cpu_offload \
          --load_tensors \
          --ignore_learned_positional_embeddings \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --allow_tf32 \
          --report_to wandb \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done

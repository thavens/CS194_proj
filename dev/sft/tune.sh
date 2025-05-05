uv run accelerate launch --num_processes 4 \
    --config_file dev/sft/deepspeed_zero3.yaml dev/sft/sft.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir /bucket/Qwen2.5-0.5B-Instruct \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --run_name 0.5B \
    --dataset_path thavens/simple_instructions \
    --dataset_subset fused \

uv run accelerate launch --num_processes 4 \
    --config_file dev/sft/deepspeed_zero3.yaml dev/sft/sft.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir /bucket/Qwen2.5-1.5B-Instruct \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --run_name 1.5B \
    --dataset_path thavens/simple_instructions \
    --dataset_subset fused \

uv run accelerate launch --num_processes 4 \
    --config_file dev/sft/deepspeed_zero3.yaml dev/sft/sft.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir /bucket/Qwen2.5-3B-Instruct \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --run_name 3B \
    --dataset_path thavens/simple_instructions \
    --dataset_subset fused \

uv run accelerate launch --num_processes 4 \
    --config_file dev/sft/deepspeed_zero3.yaml dev/sft/sft.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --output_dir /bucket/Qwen2.5-7B-Instruct \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --run_name 7B \
    --dataset_path thavens/simple_instructions \
    --dataset_subset fused \
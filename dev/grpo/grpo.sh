CUDA_VISIBLE_DEVICES=3 nohup uv run trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct &

CUDA_VISIBLE_DEVICES=0,1,2 uv run accelerate launch \
    --num_processes 3 \
    --downcast_bf16 \
    --config_file dev/sft/deepspeed_zero3.yaml dev/grpo/run.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "/bucket/Qwen2.5-7B-grpo" \
    --run_name "Qwen2.5-7B-grpo" \
    --reward_function instruction_reward
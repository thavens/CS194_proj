import datasets
from dev.grpo.reward_function import instruction_reward
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from statistics import mean
from itertools import batched

model_name = "/scratch/public_models/huggingface/Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
).to("cuda:0")
tok = AutoTokenizer.from_pretrained(
    model_name,
)

ds = datasets.load_dataset("thavens/judged_responses", split="train")
res = []
ds = ds.to_list()

for batch in batched(ds, 10):
    print("Batch size:", len(batch))
    def msgs_to_str(msgs):
        content = ""
        for msg in msgs:
            if msg["role"] == "assistant":
                content += msg["content"]
        return content

    gt = [el["new_instruction"] for el in batch]
    completions = [[{"content": msgs_to_str(el["messages"])}] for el in batch]

    scores = instruction_reward(None, completions, gt, model, tok)

    for score, el in zip(scores, batch):
        if score == el["assistant_response_judgment"]:
            res.append(1)
        else:
            res.append(0)
print(mean(res))
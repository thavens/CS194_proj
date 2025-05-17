import datasets

haiku_judged = datasets.load_from_disk("dataset_v2/judged_responses").to_list()
qwen_judged = datasets.load_from_disk("dataset_v2/qwen_judged_responses").to_list()

for h in haiku_judged:
    h["model"] = "haiku"

for q in qwen_judged:
    q["model"] = "qwen"

full = haiku_judged + qwen_judged

datasets.Dataset.from_list(full).push_to_hub("thavens/judged_responses")
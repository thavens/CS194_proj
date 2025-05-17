# This aims to fuse the examples in the dataset to improve long context performance while improving training efficiency.

# Each system prompt in the dataset is shared across three examples.
# We can take the user prompt and assistant response from those three examples and combine them into one example.

# This would assume resposes are uncorrellated and can be combined. while the system prompt is correlated and all the new instructions are independent each other.

import datasets
import re
import random
from dev.induced_instruction.utils import insert_sentence_at_random_period
import itertools

response_ds = datasets.load_dataset(
    "thavens/simple_instructions", split="train"
).to_list()
clause_ds = datasets.load_from_disk("dataset_v2/clauses").to_list()


new_inst_to_example = {ex["new_instruction"]: ex for ex in response_ds}

dataset_examples = []
for ex in clause_ds:
    matches = re.findall(r"(?s)<clause>(.*?)</clause>", ex["response"])
    clauses = [match.strip() for match in matches]

    sys_msg = ex["sys_prompt"]
    candidates = []
    for clause in clauses:
        if clause in new_inst_to_example:
            candidates.append(new_inst_to_example[clause])

        # build the fused system message
        sys_msg = insert_sentence_at_random_period(sys_msg, clause)

    # messages[0] is the system message
    # messages[1] is the user message
    # messages[2:] is the assistant message
    # extract each messages[1:]
    ua_prompts = [c["messages"][1:] for c in candidates]

    # fuse the user prompt and assistant responses randomly
    for combination in itertools.permutations(range(len(ua_prompts)), len(ua_prompts)):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        ]
        for idx in combination:
            msgs.extend(ua_prompts[idx])
        dataset_examples.append(
            {
                "messages": msgs,
                "new_instructions": ex["response"],
            }
        )

new_dataset = datasets.Dataset.from_list(dataset_examples)
new_dataset.push_to_hub("thavens/simple_instructions", "fused")

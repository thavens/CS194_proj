import torch
import random
from dev.prompts import make_judge_messages
from concurrent.futures import ThreadPoolExecutor
import openai
import dotenv

dotenv.load_dotenv()


def complete(prompt):
    if getattr(complete, "client", None) is None:
        complete.client = openai.OpenAI(
            api_key="None", base_url=f"http://localhost:{30000}/v1"
        )
        complete.model_name = complete.client.models.list().data[0].id
    response = (
        (
            complete.client.completions.create(
                model=complete.model_name,
                prompt=prompt,
                temperature=0.00,
                max_tokens=2048,
            )
        )
        .choices[0]
        .text
    )
    return response


def llm_reward(completions, ground_truth, tokenizer, **kwargs):
    prompts = [
        make_judge_messages(instruction=gt, completion=comp[0]["content"])
        for gt, comp in zip(ground_truth, completions)
    ]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            prompt, continue_final_message=True, tokenize=False
        )
        for prompt in prompts
    ]

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(complete, formatted_prompts))

    # parse the true/false response
    parsed_results = [1 if "yes" in result.lower() else 0 for result in results]
    return parsed_results


def random_reward(completions, **kwargs):
    return [random.randint(0, 1) for i in range(len(completions))]


def easy_reward(completions, **kwargs):
    return ["." not in comp[0]["content"] for comp in completions]

def reward_msgs(instruction, response):
    return [
        {"role": "user", "content": f"""\
# Context
### Rule:
{instruction}

### Response:
{response}

# Instruction
Repond with "Score: 1" if the response follows the rule.
Repond with "Score: 0" if the response does not follow the rule.

If the response follows the rule, the score is 1.
If the response does not adhere to the rule, the score is 0.
"""},
        {"role": "assistant", "content": "Score: "}
    ]

def instruction_reward(prompts, completions, ground_truth, ref_model, tokenizer):
    prompts = [
        reward_msgs(instruction=gt, response=comp[0]["content"])
        for gt, comp in zip(ground_truth, completions)
    ]
    tokenized_prompts = [
        tokenizer.apply_chat_template(prompt, continue_final_message=True)
        for prompt in prompts
    ]

    input_ids = []
    interest: list = []
    position_ids = []

    for prompt in tokenized_prompts:
        input_ids.extend(prompt)
        position_ids.extend(i for i in range(len(prompt)))
        if len(interest) == 0:
            interest.append(len(prompt) - 1)
        else:
            interest.append(interest[-1] + len(prompt))

    with torch.no_grad():
        position_ids = torch.tensor([position_ids], dtype=torch.long).cuda()
        input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
        lmoutput: torch.FloatTensor = ref_model.forward(
            input_ids, position_ids=position_ids, use_cache=False
        ).logits
    # indexes to gather
    # logits shape [batch_size, sequence_length, vocab_size]
    # indices of interest
    # " 1" token is id 16
    # " 0" token is id 15
    # [interest[0], " 1"]
    # [interest[0], " 0"]
    # [interest[1], " 1"]
    # [interest[1], " 0"]
        logits_interest: torch.Tensor = lmoutput.squeeze()[
            interest
        ]  # shape: [interest, vocab_size]

        # reverse the true, false because index 1 is true and index 0 is false based on rewards as index
        select = torch.tensor([15, 16]).cuda()
        tf_logits = logits_interest.index_select(1, select)  # shape: [interest, 2]
    return list(torch.argmax(tf_logits, dim=1).cpu())

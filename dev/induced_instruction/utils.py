import random
import re

def insert_sentence_at_random_period(base_prompt: str, sentence: str) -> str:
    # add a period to sentence if there is none before
    sentence += (
        "."
        if sentence[-1] != "." and sentence[-1] != "?" and sentence[-1] != "!"
        else ""
    )
    sentence = sentence.strip()

    # Find all period positions in the base prompt using regex
    period_matches = list(re.finditer(r"\.", base_prompt))

    # If there are no periods, just append the sentence
    if not period_matches:
        return f"{base_prompt}\n{sentence}"

    # Choose a random period match
    chosen_match = random.choice(period_matches)
    start, end = chosen_match.span()

    # Construct the new prompt
    new_prompt = base_prompt[:end] + f" {sentence}" + base_prompt[end:]
    return new_prompt

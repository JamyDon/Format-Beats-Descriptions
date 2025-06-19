from typing import List, Tuple


TEMPLATE2FUNC = {}

def get_prompt_generator(template_name: str):
    if template_name not in TEMPLATE2FUNC:
        raise NotImplementedError(f"Template {template_name} is not registered.")
    return TEMPLATE2FUNC[template_name]

def template_register(func):
    TEMPLATE2FUNC[func.__name__.replace("template_", "")] = func
    return func

@template_register
def template_vanilla(instruction: str, test_input: str, examples: List[str], shot: int=4, **kwargs) -> str:
    prompt = []

    prompt.append(f'Instruction: {instruction}')

    for example in examples[:shot]:
        prompt.append(example + "\n")

    prompt.append(test_input)

    prompt = "\n".join(prompt)
    return prompt

@template_register
def template_ensemble_random(instruction, test_input: str, examples: List[str], before_test_input: str, rand_nouns: Tuple[str], shot: int=4) -> str:
    num_group = len(rand_nouns)
    if shot % num_group != 0:
        raise ValueError("Number of examples should be divisible by the number of random nouns.")
    num_per_group = shot // num_group
    
    prompt = []

    prompt.append(f'Instruction: {instruction}')

    for i in range(0, shot, num_per_group):
        prompt.append(f'Examples with similar {rand_nouns[i // num_per_group]}:')
        for example in examples[i:i+num_per_group]:
            prompt.append(example + "\n")

    prompt.append(before_test_input)
    prompt.append(test_input)

    prompt = "\n".join(prompt)
    return prompt

@template_register
def template_ensemble_different_random(instruction, test_input: str, examples: List[str], before_test_input: str, rand_nouns: Tuple[str], shot: int=4) -> str:
    num_group = len(rand_nouns)
    if shot % num_group != 0:
        raise ValueError("Number of examples should be divisible by the number of random nouns.")
    num_per_group = shot // num_group
    
    prompt = []

    prompt.append(f'Instruction: {instruction}')

    for i in range(0, shot, num_per_group):
        prompt.append(f'Examples with different {rand_nouns[i // num_per_group]}:')
        for example in examples[i:i+num_per_group]:
            prompt.append(example + "\n")

    prompt.append(before_test_input)
    prompt.append(test_input)

    prompt = "\n".join(prompt)
    return prompt

import dataclasses
from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval import instructions_registry

import nltk
nltk.download('punkt')  # Download the punkt tokenizer, which includes French
nltk.download('punkt_tab')

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def filter_not_null_entries_all(input_dict):
    # Filter the dictionary to include all non-null entries
    non_null_entries = {k: v for k, v in input_dict.items() if v is not None}
    non_null_entries_arr = [{k: v} for k, v in input_dict.items() if v is not None]

    # If both 'section_spliter' and 'num_sections' are in the dictionary, combine them
    if 'section_spliter' in non_null_entries and 'num_sections' in non_null_entries:
        non_null_entries
        combined_section = {
            'section_spliter': non_null_entries.pop('section_spliter'),
            'num_sections': non_null_entries.pop('num_sections')
        }
        # List of keys we want to remove
        keys_to_remove = ['section_spliter', 'num_sections']

        # Clean the dictionaries using a for loop
        cleaned_data = []
        for entry in non_null_entries_arr:
            # Create a copy of the dictionary without the unwanted keys
            cleaned_entry = {k: v for k, v in entry.items() if k not in keys_to_remove}
            if cleaned_entry:  # Only add non-empty dictionaries
                cleaned_data.append(cleaned_entry)
        non_null_entries_arr.append(combined_section)
        return cleaned_data

    return non_null_entries_arr
    
def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        #results=[parse_kwargs_list(inp.kwargs)]
        # Remove the last three elements
        keys = list(inp.kwargs.keys())[:-3]
        trimmed_kwargs = {k: inp.kwargs[k] for k in keys}

        result = filter_not_null_entries_all(trimmed_kwargs)
        try:
            #kwargs = {k: v for k, v in result[0].items() if v}
            instruction.build_description(**result[0])
        except:
                try:
                    #kwargs = {k: v for k, v in result[1].items() if v}
                    instruction.build_description(**result[1])
                except:
                    try:
                        #kwargs = {k: v for k, v in result[2].items() if v}
                        instruction.build_description(**result[2])
                    except:
                        instruction.build_description()



        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
def parse_kwargs_list(kwargs_list):
    parsed_kwargs = {}
    non_null_instances = []

    # Parse the list of 'key:value' strings into a dictionary
    for pair in kwargs_list:
        if ":" in pair:
            k, v = pair.split(":", 1)
            parsed_kwargs[k] = None if v == "None" else v

    # Split the dictionary into separate dictionaries based on non-null values
    current_dict = {}
    for k, v in parsed_kwargs.items():
        if v is not None:
            current_dict[k] = v
        # If current_dict is not empty, store it in the list and reset
        if len(current_dict) > 0:
            non_null_instances.append(current_dict)
            current_dict = {}

    # Reverse the list of non-null instances before returning
    return non_null_instances



def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        keys = list(inp.kwargs.keys())[:-3]
        trimmed_kwargs = {k: inp.kwargs[k] for k in keys}
        result = filter_not_null_entries_all(trimmed_kwargs)
        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        try:
            kwargs = {k: v for k, v in result[0].items() if v}
            instruction.build_description(**kwargs)
        except:
                try:
                    kwargs = {k: v for k, v in result[1].items() if v}
                    instruction.build_description(**kwargs)
                except:
                    try:
                        kwargs = {k: v for k, v in result[2].items() if v}
                        instruction.build_description(**kwargs)
                    except:
                        instruction.build_description()
        
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=0,
        instruction_id_list=doc["categories"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]
    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc

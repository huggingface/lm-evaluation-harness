#!/usr/bin/env python

"""
Normalize benchmark scores in le-leadboard/OpenLLMFrenchLeaderboard as in HF Open LLM Leaderboard v2.

Adapted from https://colab.research.google.com/drive/1-aPrFJjwdifhVLxzJcsYXeebqNi_5vaw?usp=sharing#scrollTo=z7npCF8086XG

Usage:
    python scripts/normalize_scores.py <path/to/lm_eval/output.json>
"""


import json

import fire
import numpy as np


# Normalization function
def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100


bbh_subtasks = {
    "compréhension_des_sports": 2,
    "suivi_objets_mélangés_trois_objets": 3,
    "naviguer": 2,
    "sarcasmes": 2,
    "compréhension_de_la_date": 6,
    "raisonnement_sur_les_objets_colorés": 18,
    "comptage_d_objets": 19,
    "déduction_logique_sept_objets": 7,
    "formes_géométriques": 11,
    "toile_de_mensonges": 2,
    "recommandation_de_film": 6,
    "déduction_logique_cinq_objets": 5,
    "détection_d'erreurs_de_traduction_sailantes": 6,
    "désambiguïsation_qa": 3,
    "séquences_temporelles": 4,
    "hyperbate": 2,
    "déduction_logique_trois_objets": 3,
    "jugement_causal": 2,
    "sophismes_formels": 2,
    "suivi_objets_mélangés_sept_objets": 7,
    # "ruin_names": 6,  # todo
    "pingouins_sur_une_table": 5,
    "expressions_booléennes": 2,
    "suivi_objets_mélangés_cinq_objets": 5,
}

musr_subtasks = {"murder_mysteries_fr": 2, "object_placements_fr": 5, "team_allocation_fr": 3}

math_subtasks = ["algebra", "counting_and_prob", "geometry", "intermediate_algebra", "num_theory", "prealgebra", "precalculus"]

gpqa_subtasks = ["diamond", "extended", "main"]


def main(input_file: str):
    # 1. Load the lm_eval output json file
    with open(input_file, "r") as file:
        data = json.load(file)

    # 2. Extract model name and precision
    model_name = data["model_name"]
    # model_dtype = data["config"]["model_dtype"]
    # model_revision = data["config"]["model_revision"]

    # 3. Normalize tasks without subtasks
    # 3.1. Normalize MMMLU-fr scores
    mmlu_raw_score = data["results"]["leaderboard_mmlu_fr"]["acc,none"]
    mmlu_score = normalize_within_range(mmlu_raw_score, 0.1, 1.0)

    # 3.2. Normalize GPQA scores
    gpqa_scores = []
    for subtask in gpqa_subtasks:
        subtask_key = f"leaderboard_gpqa_{subtask}_fr"
        if subtask_key in data["results"]:
            gpqa_raw_score = data["results"][subtask_key]["acc_norm,none"]
            gpqa_score = normalize_within_range(gpqa_raw_score, 0.25, 1.0)
            gpqa_scores.append(gpqa_score)
        else:
            print(f"Missing result for subtask: {subtask_key}")
    # Average GPQA score
    gpqa_score = sum(gpqa_scores) / len(gpqa_scores)

    # 4. Normalize tasks with subtasks
    # 4.1. Normalize BBH scores
    bbh_scores = []
    for subtask, num_choices in bbh_subtasks.items():
        subtask_key = f"leaderboard_bbh_{subtask}"
        if subtask_key in data["results"]:
            bbh_raw_score = data["results"][subtask_key]["acc_norm,none"]
            lower_bound = 1 / num_choices
            bbh_score = normalize_within_range(bbh_raw_score, lower_bound, 1.0)
            bbh_scores.append(bbh_score)
        else:
            print(f"Missing result for subtask: {subtask_key}")
    # Average BBH score
    bbh_score = sum(bbh_scores) / len(bbh_scores)

    # 4.2. Normalize MUSR scores
    musr_scores = []
    for subtask, num_choices in musr_subtasks.items():
        subtask_key = f"leaderboard_musr_{subtask}"
        if subtask_key in data["results"]:
            musr_raw_score = data["results"][subtask_key]["acc_norm,none"]
            lower_bound = 1 / num_choices
            musr_score = normalize_within_range(musr_raw_score, lower_bound, 1.0)
            musr_scores.append(musr_score)
        else:
            print(f"Missing result for subtask: {subtask_key}")
    # Average MUSR scores
    musr_score = sum(musr_scores) / len(musr_scores)

    # 5. Generative evaluations
    # 5.1. Compute MATH-Hard scores
    math_scores = []
    for subtask in math_subtasks:
        subtask_key = f"leaderboard_math_{subtask}_hard_fr"
        if subtask_key in data["results"]:
            math_raw_score = data["results"][subtask_key]["exact_match,none"]
            math_score = normalize_within_range(math_raw_score, 0, 1.0)
            math_scores.append(math_score)
        else:
            print(f"Missing result for subtask: {subtask_key}")
    # Average MATH-Hard score
    math_score = sum(math_scores) / len(math_scores)

    # 5.2. Compute IFEval scores
    ifeval_inst_score = data["results"]["leaderboard_ifeval_fr"]["inst_level_strict_acc,none"] * 100
    ifeval_prompt_score = data["results"]["leaderboard_ifeval_fr"]["prompt_level_strict_acc,none"] * 100
    # Average IFEval score
    ifeval_score = (ifeval_inst_score + ifeval_prompt_score) / 2

    # 6. Calculate overall score
    overall_score = (bbh_score + math_score + gpqa_score + mmlu_score + musr_score + ifeval_score) / 6

    # Round all scores to 2 decimal places
    bbh_score = round(bbh_score, 2)
    math_score = round(math_score, 2)
    gpqa_score = round(gpqa_score, 2)
    mmlu_score = round(mmlu_score, 2)
    musr_score = round(musr_score, 2)
    ifeval_score = round(ifeval_score, 2)
    overall_score = round(overall_score, 2)

    # 7. Export
    # results = {
    #     "Model name": model_name,
    #     # "Precision": precision,
    #     # "Revision": revision,
    #     "IFEval": ifeval_score,
    #     "BBH": bbh_score,
    #     "MATH-lvl5": math_score,
    #     "GPQA": gpqa_score,
    #     "MUSR": musr_score,
    #     "MMMLU-fr": mmlu_score,
    # }
    results = {
        "config": {
            "model_name": model_name,
             "model_dtype": "torch.float16",
        },
        "results": {
            "BBH-fr": {
                "metric_name": bbh_score/100,
            },
            "GPQA-fr": {
                "metric_name": gpqa_score/100,
            },
            "IFEval-fr": {
                "metric_name": ifeval_score/100, 
            },
            "MUSR-fr": {
                "metric_name": musr_score/100,
            },
            "MATH Lvl5-fr": {
                "metric_name": math_score/100,
            },
            "MMMLU-fr": {
                "metric_name": mmlu_score/100,
            },
        },
    }

    print(json.dumps(results, indent=4))

    output_file = f"{input_file[:-5]}_norm.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)

# Leaderboard evaluations
Our goal with this group is to diversify the stable sets by enabling them to a new category of models Namely, Multilingual Models and in process create an unchanging through time version of evaluations that will power the Open LLM French Leaderboard on HuggingFace.

As we want to evaluate models across capabilities, the list currently contains:
- BBH-fr (3-shots, multichoice)
- GPQA-fr (0-shot, multichoice)
- mmmlu-fr (5-shots, multichoice)
- Musr-fr (0-shot, multichoice)
- ifeval-fr (0-shot, generative)
- Math-lvl-5-fr (4-shots, generative, minerva version)


Details on the choice of those evals can be found [here]([https://huggingface.co/spaces/open-llm-leaderboard/blog](https://huggingface.co/le-leadboard)) !

## BigBenchHard (BBH)

A suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH).
These are the task for which prior language model evaluations did not
outperform the average human-rater.

### Paper

Title: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them

BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models?
In this work, we focus on a suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH). These are the task for which prior language model evaluations did not outperform the average human-rater. We find that applying chain-of-thought (CoT) prompting to BBH tasks enables PaLM to surpass the average human-rater performance on 10 of the 23 tasks, and Codex (code-davinci-002) to surpass the average human-rater performance on 17 of the 23 tasks. Since many tasks in BBH require multi-step reasoning, few-shot prompting without CoT, as done in the BIG-Bench evaluations (Srivastava et al., 2022), substantially underestimates the best performance and capabilities of language models, which is better captured via CoT prompting. As further analysis, we explore the interaction between CoT and model scale on BBH, finding that CoT enables emergent task performance on several BBH tasks with otherwise flat scaling curves.


- paper: https://huggingface.co/papers/2210.09261
- Homepage: https://github.com/suzgunmirac/BIG-Bench-Hard

### Citation

```
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
```

### Groups

- `leaderboard_bbh_fr`

### Tasks

- `leaderboard_bbh_boolean_expressions_fr`
- `leaderboard_bbh_causal_judgement_fr`
- `leaderboard_bbh_date_understanding_fr`
- `leaderboard_bbh_disambiguation_qa_fr`
- `leaderboard_bbh_dyck_languages_fr`
- `leaderboard_bbh_formal_fallacies_fr`
- `leaderboard_bbh_geometric_shapes_fr`
- `leaderboard_bbh_hyperbaton_fr`
- `leaderboard_bbh_logical_deduction_five_objects_fr`
- `leaderboard_bbh_logical_deduction_seven_objects_fr`
- `leaderboard_bbh_logical_deduction_three_objects_fr`
- `leaderboard_bbh_movie_recommendation_fr`
- `leaderboard_bbh_multistep_arithmetic_two_fr`
- `leaderboard_bbh_navigate_fr`
- `leaderboard_bbh_object_counting_fr`
- `leaderboard_bbh_penguins_in_a_table_fr`
- `leaderboard_bbh_reasoning_about_colored_objects_fr`
- `leaderboard_bbh_ruin_names_fr`
- `leaderboard_bbh_salient_translation_error_detection_fr`
- `leaderboard_bbh_snarks_fr`
- `leaderboard_bbh_sports_understanding_fr`
- `leaderboard_bbh_temporal_sequences_fr`
- `leaderboard_bbh_tracking_shuffled_objects_five_objects_fr`
- `leaderboard_bbh_tracking_shuffled_objects_seven_objects_fr`
- `leaderboard_bbh_tracking_shuffled_objects_three_objects_fr`
- `leaderboard_bbh_web_of_lies_fr`
- `leaderboard_bbh_word_sorting_fr`

## GPQA

### Paper

Title: GPQA: A Graduate-Level Google-Proof Q&A Benchmark

We present GPQA, a challenging dataset of 448 multiple-choice questions written
by domain experts in biology, physics, and chemistry. We ensure that the
questions are high-quality and extremely difficult: experts who have or are
pursuing PhDs in the corresponding domains reach 65% accuracy (74% when
discounting clear mistakes the experts identified in retrospect), while highly
skilled non-expert validators only reach 34% accuracy, despite spending on
average over 30 minutes with unrestricted access to the web (i.e., the
questions are “Google-proof”). The questions are also difficult for
state-of-the-art AI systems, with our strongest GPT-4–based baseline achieving
39% accuracy. If we are to use future AI systems to help us answer very hard
questions—for example, when developing new scientific knowledge—we need to
develop scalable oversight methods that enable humans to supervise their
outputs, which may be difficult even if the supervisors are themselves skilled
and knowledgeable. The difficulty of GPQA both for skilled non-experts and
frontier AI systems should enable realistic scalable oversight experiments,
which we hope can help devise ways for human experts to reliably get truthful
information from AI systems that surpass human capabilities.

- Paper: https://huggingface.co/papers/2311.12022
- Homepage: https://github.com/idavidrein/gpqa/tree/main

### Citation

```
@misc{rein2023gpqa,
      title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      year={2023},
      eprint={2311.12022},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

### Groups

- `leaderboard_gpqa_fr`

### Tasks

- `leaderboard_gpqa_extended_fr`
- `leaderboard_gpqa_diamond_fr`
- `leaderboard_gpqa_main_fr`

## IFEval

### Paper

Title: Instruction-Following Evaluation for Large Language Models

One core capability of Large Language Models (LLMs) is to follow natural
language instructions. However, the evaluation of such abilities is not
standardized: Human evaluations are expensive, slow, and not objectively
reproducible, while LLM-based auto-evaluation is potentially biased or limited
by the ability of the evaluator LLM. To overcome these issues, we introduce
Instruction-Following Eval (IFEval) for large language models. IFEval is a
straightforward and easy-to-reproduce evaluation benchmark. It focuses on a set
of "verifiable instructions" such as "write in more than 400 words" and
"mention the keyword of AI at least 3 times". We identified 25 types of those
verifiable instructions and constructed around 500 prompts, with each prompt
containing one or more verifiable instructions. We show evaluation results of
two widely available LLMs on the market.

- Paper: https://huggingface.co/papers/2210.09261
- Homepage: https://github.com/google-research/google-research/tree/master/instruction_following_eval

### Citation

```
@article{zhou2023instructionfollowing,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023},
}
```

### Tasks

- `leaderboard_ifeval_fr`

## MATH-hard

This is the 4 shots variant of minerva math but only keeping the level 5 questions.

### Paper

Title: Measuring Mathematical Problem Solving With the MATH Dataset

Many intellectual endeavors require mathematical problem solving, but this
skill remains beyond the capabilities of computers. To measure this ability in
machine learning models, we introduce MATH, a new dataset of 12,500 challenging
competition mathematics problems. Each problem in MATH has a full step-by-step
solution which can be used to teach models to generate answer derivations and
explanations.

NOTE: The few-shot and the generated answer extraction is based on the
[Minerva](https://arxiv.org/abs/2206.14858) and exact match equivalence is
calculated using the `sympy` library. This requires additional dependencies,
which can be installed via the `lm-eval[math]` extra.

- Paper: https://huggingface.co/papers/2103.03874
- Homepage: https://github.com/hendrycks/math


### Citation

```
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
@misc{2206.14858,
Author = {Aitor Lewkowycz and Anders Andreassen and David Dohan and Ethan Dye and Henryk Michalewski and Vinay Ramasesh and Ambrose Slone and Cem Anil and Imanol Schlag and Theo Gutman-Solo and Yuhuai Wu and Behnam Neyshabur and Guy Gur-Ari and Vedant Misra},
Title = {Solving Quantitative Reasoning Problems with Language Models},
Year = {2022},
Eprint = {arXiv:2206.14858},
}
```

### Groups

- `leaderboard_math_hard_fr`

### Tasks

- `leaderboard_math_algebra_hard_fr`
- `leaderboard_math_counting_and_prob_hard_fr`
- `leaderboard_math_geometry_hard_fr`
- `leaderboard_math_intermediate_algebra_hard_fr`
- `leaderboard_math_num_theory_hard_fr`
- `leaderboard_math_prealgebra_hard_fr`
- `leaderboard_math_precalc_hard_fr`


## MMMLU-fr

### Model Card

MMMLU-fr is the french split of the dataset openai/MMMLU

Multilingual Massive Multitask Language Understanding (MMMLU)
The MMLU is a widely recognized benchmark of general knowledge attained by AI models. It covers a broad range of topics from 57 different categories, covering elementary-level knowledge up to advanced professional subjects like law, physics, history, and computer science.

We translated the MMLU’s test set into 14 languages using professional human translators. Relying on human translators for this evaluation increases confidence in the accuracy of the translations, especially for low-resource languages like Yoruba. We are publishing the professional human translations and the code we use to run the evaluations.

This effort reflects our commitment to improving the multilingual capabilities of AI models, ensuring they perform accurately across languages, particularly for underrepresented communities. By prioritizing high-quality translations, we aim to make AI technology more inclusive and effective for users worldwide.

- Paper:  https://arxiv.org/abs/2009.03300
- Homepage: https://github.com/openai/simple-evals

### Citation

```
@misc{wang2024mmluprorobustchallengingmultitask,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language
      Understanding Benchmark},
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01574},
}
```

### Groups

- `leaderboard_mmlu_fr`

### Tasks

- `leaderboard_mmlu_fr`


## Musr

### Paper

Title: MuSR: Testing the Limits of Chain-of-thought with Multistep Soft
Reasoning  

While large language models (LLMs) equipped with techniques like
chain-of-thought prompting have demonstrated impressive capabilities, they
still fall short in their ability to reason robustly in complex settings.
However, evaluating LLM reasoning is challenging because system capabilities
continue to grow while benchmark datasets for tasks like logical deduction have
remained static. We introduce MuSR, a dataset for evaluating language models on
multistep soft reasoning tasks specified in a natural language narrative. This
dataset has two crucial features. First, it is created through a novel
neurosymbolic synthetic-to-natural generation algorithm, enabling the
construction of complex reasoning instances that challenge GPT-4 (e.g., murder
mysteries roughly 1000 words in length) and which can be scaled further as more
capable LLMs are released. Second, our dataset instances are free text
narratives corresponding to real-world domains of reasoning; this makes it
simultaneously much more challenging than other synthetically-crafted
benchmarks while remaining realistic and tractable for human annotators to
solve with high accuracy. We evaluate a range of LLMs and prompting techniques
on this dataset and characterize the gaps that remain for techniques like
chain-of-thought to perform robust reasoning.

- Paper: https://huggingface.co/papers/2310.16049
- Homepage: https://zayne-sprague.github.io/MuSR/

### Citation

```
@misc{sprague2024musrtestinglimitschainofthought,
      title={MuSR: Testing the Limits of Chain-of-thought with Multistep Soft
      Reasoning},
      author={Zayne Sprague and Xi Ye and Kaj Bostrom and Swarat Chaudhuri and Greg Durrett},
      year={2024},
      eprint={2310.16049},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.16049},
}
```

### Groups

- `leaderboard_musr_fr`

### Tasks

- `leaderboard_musr_murder_mysteries`
- `leaderboard_musr_object_placements`
- `leaderboard_musr_team_allocation`

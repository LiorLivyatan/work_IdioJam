"""
Utils for the Multi-word Expression Identification (MWEI) task.
"""

###############################################################################
# Imports
import os
import pandas as pd
import json
from typing import Callable
from pydantic import BaseModel
import random
import ast
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.utils import FEW_SHOT_PROMPT_TEMPLATE, MERGE_COLUMNS
from src.coam_utils import self_consistency

###############################################################################

###############################################################################
# Constants
TASK_COLUMNS = MERGE_COLUMNS.copy()

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(parent_dir, "data/coam")

###############################################################################

###############################################################################

PROMPTS = {
    "system": """You are a helpful system to identify multiple-word expressions (MWEs).
Identify all the MWEs in the given sentence, and output their surface forms and the indices of their components. Here, an MWE is defined as a sequence that satisfies the following three conditions.
1. It consists of multiple words that are always realized by the same lexemes. Such words cannot be replaced without distorting the meaning of the expression or violating language conventions.
2. It displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs when the meaning of an expression cannot be explicitly derived from its components. In other words, a semantically idiomatic takes on a meaning that is unique to that combination of words. Lexical idiomaticity occurs when one or more components of an expression are not used as stand-alone words in standard English. Syntactic idiomaticity occurs when the grammar of an expression cannot be derived directly from that of its components. For example, semantically idiomatic MWEs include "break up", the lexically idiomatic include "to and from", and the syntactically idiomatic include "long time no see".
3. It is not a multi-word named entity, i.e., a specific name of a person, facility, etc.
Output the sentence in TSV format, where each row contains a word and its MWE tag. Assign the same numeric tag (starting from 1) to all words that belong to the same MWE. Use 0 if the word is not part of any MWE. If a word belongs to multiple MWEs, concatenate tags with semicolons (e.g., 2;5). Ensure that all words in a valid MWE are tagged, even if they appear in separate lines. Include the first word of the MWE, not just the idiomatic or fixed component.""",
    "user": "Sentence:{sentence}",
}


###############################################################################


###############################################################################
# Functions


def get_data(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for the given task and prompt type.
    :param kwargs: keyword arguments
    :return: data
    """

    # Get data
    test_dir = os.path.join(DATA_DIR, "testset.tsv")
    train_dir = os.path.join(DATA_DIR, "trainset.tsv")
    test = pd.read_csv(test_dir, sep="\t")
    train = pd.read_csv(train_dir, sep="\t")

    return train, test


def safe_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            print(f"Error parsing: {x}, error: {e}")
            return []
    return x


def generate_mwe_tsv(tokens_clean, mwes):
    # Initialize all tokens as "0"
    tags = ["0"] * len(tokens_clean)

    for mwe_id, mwe in enumerate(mwes, start=1):
        for idx in mwe["indices"]:
            if tags[idx] == "0":
                tags[idx] = str(mwe_id)
            else:
                tags[idx] += f";{mwe_id}"

    # Build TSV string
    lines = ["Word\tMWE_Tag"]
    for token, tag in zip(tokens_clean, tags):
        lines.append(f"{token}\t{tag}")

    return "```\ntsv\n" + "\n".join(lines) + "\n```"


def get_few_shot_prompt(
    train: pd.DataFrame, seed, prompt_type: str, shots: int = 6,
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param train: training data
    :param lang: language
    :param seed: seed for reproducibility
    :param prompt_type: prompt type
    :param shots: number of few-shot examples
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility
    if "full" in prompt_type:
        raise NotImplementedError(
            "This prompt type is not implemented. Please use few-shot or zero-shot prompt type."
        )
    else:
        shots = int(shots / 2)
        # Get 3 examples with some wme and 3 examples without mwe
        # surface is the true idiom
        # Select examples with MWE (non-empty surface list)
        train["tokens_clean"] = train["tokens_clean"].apply(safe_parse)
        train["mwes"] = train["mwes"].apply(safe_parse)
        train["surface"] = train["surface"].apply(safe_parse)

        shots_with_mwe = train[train["mwes"].apply(lambda x: len(x) > 0)].sample(
            shots, random_state=seed
        )

        # Select examples without MWE (empty surface list)
        shots_without_mwe = train[train["mwes"].apply(lambda x: len(x) < 1)].sample(
            shots, random_state=seed
        )

        few_shot_examples = pd.DataFrame()

        # Combine the examples
        few_shot_examples = pd.concat(
            [shots_with_mwe, shots_without_mwe], ignore_index=True
        )

        examples = [
            {
                "input": f"Sentence: \n{"\n".join(row['tokens_clean'])}",
                "output": json.dumps({"mwes": row["mwes_final"]}),
            }
            for _, row in few_shot_examples.iterrows()
        ]

    # Shuffle the examples
    random.shuffle(examples)  # Shuffle the list in place

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=examples,
    )
    return few_shot_prompt


def get_prompt_schema(config: dict, **kwargs) -> tuple[ChatPromptTemplate, None]:
    """
    Get the prompt for the given task and prompt type.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt and schema
    """
    # Get kwargs
    train = kwargs["train"]
    prompt_type = config["prompt_type"]
    seed = config["seed"]
    shots = config["shots"]
    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = get_few_shot_prompt(
            train, seed=seed, prompt_type=prompt_type, shots=shots
        )
        # Add in position 1 (after the system prompt)
        messages.insert(1, few_shot_prompt)

    prompt = ChatPromptTemplate.from_messages(messages)


    return prompt, None


def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs
    """
    data["tokens_clean"] = data["tokens_clean"].apply(safe_parse)
    user_inputs = [
        {"sentence": "\n".join(row["tokens_clean"])} for _, row in data.iterrows()
    ]
    return user_inputs


def extract_mwes_from_tsv(tsv_string):
    lines = tsv_string.strip().splitlines()
    word_tag_pairs = []

    # Skip header if present
    lines = tsv_string.strip().splitlines()
    start_index = 0
    if lines[0].startswith("Word") or lines[0].startswith("```"):
        if lines[1].startswith("Word") or lines[1].startswith("```"):
            start_index = 2
        else:
            start_index = 1

    # Parse word-tag pairs
    for line in lines[start_index:]:
        if line.startswith("```"):  # Handle closing ```
            continue
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        word, tag = parts
        word_tag_pairs.append((word, tag))

    # Group words by MWE tags
    from collections import defaultdict

    tag_to_words = defaultdict(list)

    for idx, (word, tag) in enumerate(word_tag_pairs):
        if tag == "0":
            continue
        for subtag in tag.split(";"):
            tag_to_words[subtag].append((idx, word))

    # Sort by appearance and extract surface forms
    _mwe = []
    for tag in tag_to_words.keys():
        if tag in {"MWE_Tag", "0"}:
            continue  # Skip this tag
        words = [w for _, w in sorted(tag_to_words[tag])]
        mwe = " ".join(words)
        _mwe.append(mwe)

    return _mwe


def process_responses(
    results: list[dict], test: pd.DataFrame, calc_metrics: Callable, **kwargs
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate metrics for the given task after post-processing the results.
    :param results: results
    :param test: test data
    :return: metrics and test data after post-processing and the csv formatted metrics
    """
    predicted_mwe = []
    ratios = []
    hallucinated = []
    hallucination_nr = 0
    test["predicted_mwe"] = [[] for _ in range(len(test))]

    # Calculate score
    for i, res in enumerate(results):
        # Collect PIE from all responses (Self-Consistency)
        runs_mwe = []
        at_least_one_resp_with_no_hallucination = False
        for resp in res["responses"]:
            try:
                if "content" not in resp:
                    # If the response is not in the expected format, skip it
                    _mwe = []
                else:
                    content = resp["content"]
                    # Collect the mwes from the response
                    try:
                        _mwe = extract_mwes_from_tsv(content)
                    except Exception as e:
                        _mwe = []
                    # Make sure _mwe is a list of strings - use eval
                    if isinstance(_mwe, str):
                        _mwe = eval(_mwe)
                    at_least_one_resp_with_no_hallucination = True
                runs_mwe.append(_mwe)

            except Exception as e:
                # If the response is not in the expected format, skip it
                print(
                    f"Error parsing response: {e}: Unexpected response format: {resp['parsed']}"
                )
                hallucination_nr += 1
                continue

        # If no response was valid, skip this sentence
        if not at_least_one_resp_with_no_hallucination:
            predicted_mwe.append([])
            ratios.append([])
            hallucinated.append(True)
        else:
            # Apply self-consistency to get the most common mwes
            best_mwe_with_ratio = self_consistency(runs_mwe)
            best_mwe = list(best_mwe_with_ratio.keys())
            best_ratios = list(best_mwe_with_ratio.values())
            predicted_mwe.append(best_mwe)
            ratios.append(best_ratios)
            hallucinated.append(False)

    # Add mwe to test data and their ratios
    test["predicted_mwe"] = predicted_mwe
    test["ratios"] = ratios
    test["hallucinated"] = hallucinated

    metrics = {}
    # Collect all labels and predicted tags
    log_cm_report = {}

    kwargs = {
        "gold_col": "surface",
        "pred_col": "predicted_mwe",
        "tokenized": False,
        "parseme": False,
    }

    metrics = calc_metrics(test, **kwargs)

    # Create a DataFrame with the metrics for CSV based on the res_columns
    csv_metrics = pd.DataFrame(columns=TASK_COLUMNS)
    for key in metrics.keys():
        csv_metrics.loc[0, f"{key}_mwe_precision"] = metrics[key]["mwe_based"]["P"]
        csv_metrics.loc[0, f"{key}_mwe_recall"] = metrics[key]["mwe_based"]["R"]
        csv_metrics.loc[0, f"{key}_mwe_f1"] = metrics[key]["mwe_based"]["F1"]
        csv_metrics.loc[0, f"{key}_token_precision"] = metrics[key]["token_based"]["P"]
        csv_metrics.loc[0, f"{key}_token_recall"] = metrics[key]["token_based"]["R"]
        csv_metrics.loc[0, f"{key}_token_f1"] = metrics[key]["token_based"]["F1"]

    metrics["hallucinations"] = int(test["hallucinated"].sum())
    csv_metrics.loc[0, "hallucinations"] = metrics["hallucinations"]
    log_cm_report = 0


    return metrics, test, csv_metrics, log_cm_report


###############################################################################

###############################################################################
# Export
COAM_TSV_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}

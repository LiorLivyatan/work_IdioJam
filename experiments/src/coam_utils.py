"""
Utils for the Multi-word Expression Identification (MWEI) task.
"""

###############################################################################
# Imports
import os
import json
import pandas as pd
from typing import Callable
from pydantic import BaseModel
import random
import ast
from collections import Counter
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.utils import FEW_SHOT_PROMPT_TEMPLATE, MERGE_COLUMNS, clean_predictions, make_examples
from src.pydantic_schemas import PYDANTIC_SCHEMAS
from src.typed_schemas import TYPED_SCHEMAS

###############################################################################

###############################################################################
# Constants
TASK_COLUMNS = MERGE_COLUMNS.copy()

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(parent_dir, "data/coam")

###############################################################################

###############################################################################

# promt type 6 -
PROMPTS = {
    "system": """You are a helpful system to identify multiple-word expressions (MWEs).
Identify all the MWEs in the given sentence, and output their surface forms exactly as they appear.

Here, an MWE is defined as a sequence that satisfies the following three conditions.\n
1. It consists of multiple words that are always realized by the same lexemes. Such words
cannot be replaced without distorting the meaning of the expression or violating language
conventions.\n
2. It displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs
when the meaning of an expression cannot be explicitly derived from its components. In
other words, a semantically idiomatic takes on a meaning that is unique to that combination
of words. Lexical idiomaticity occurs when one or more components of an expression are not
used as stand-alone words in standard English. Syntactic idiomaticity occurs when the
grammar of an expression cannot be derived directly from that of its components. For
example, semantically idiomatic MWEs include "break up", the lexically idiomatic include
"to and from", and the syntactically idiomatic include "long time no see".\n
3. It is not a multi-word named entity, i.e., a specific name of a person, facility, etc.

Additional instructions:
- Be cautious: Only identify an expressions as MWEs if they clearly satisfies the conditions above.
- When listing MWEs, use exactly the original surface form as it appears in the sentence.
- Only answer in JSON.""",
    "user": "Sentence:{sentence}\n",
}


###############################################################################


###############################################################################
# Functions

def extract_mwes(row):
    mwe_list = []
    mwe_tokenized_list = []

    for mwe in row["mwes"]:
        type_mwe = mwe["type"]
        surface = mwe["surface"]
        indices = mwe["indices"]
        mwe_list.append((type_mwe, surface))

        mwe_tokenized_list.append((type_mwe, [row["tokens"][i]["surface"] for i in indices]))


    return {
        "surface": mwe_list,
        "surface_tokens": mwe_tokenized_list,
    }

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

    # Change text column to sentence
    test.rename(columns={"text": "sentence"}, inplace=True)
    train.rename(columns={"text": "sentence"}, inplace=True)

    return train, test


def safe_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            print(f"Error parsing: {x}, error: {e}")
            return []
    return x


def get_few_shot_prompt(
    train: pd.DataFrame, seed, prompt_type: str, schemas, task: str, shots: int = 6, surface=True
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param train: training data
    :param lang: language
    :param seed: seed for reproducibility
    :param prompt_type: prompt type
    :param shots: number of few-shot examples
    :param surface: if True, use surface forms, else use spans
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
        train["mwes"] = train["mwes"].apply(safe_parse)
        train["surface"] = train["surface"].apply(safe_parse)

        shots_with_mwe = train[train["mwes"].apply(lambda x: len(x) > 0)].sample(
            shots, random_state=seed
        )

        # Select examples without MWE (empty surface list)
        shots_without_mwe = train[train["mwes"].apply(lambda x: len(x) < 1)].sample(
            shots, random_state=seed
        )

        shots_without_mwe["mwes_final"] = [[] for _ in range(len(shots_without_mwe))]
        few_shot_examples = pd.DataFrame()
        if surface:
            shots_with_mwe["mwes_final"] = shots_with_mwe["surface"].apply(
                lambda x: [item[1] for item in x]
            )
            # Combine the examples
            few_shot_examples = pd.concat(
                [shots_with_mwe, shots_without_mwe], ignore_index=True
            )

        else:
            shots_with_mwe["mwes_final"] = shots_with_mwe["surface"].apply(
                lambda x: [item[2] for item in x]
            )
            few_shot_examples = pd.concat(
                [shots_with_mwe, shots_without_mwe], ignore_index=True
            )

    examples = make_examples(few_shot_examples, schemas, schema_type=prompt_type, task=task)
            

    # Shuffle the examples
    random.shuffle(examples)  # Shuffle the list in place

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=examples,
    )
    return few_shot_prompt


def get_prompt_schema(config: dict, **kwargs) -> tuple[ChatPromptTemplate, BaseModel]:
    """
    Get the prompt for the given task and prompt type.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt and schema
    """
    # long_prompt_file = "src/coam_prompt.txt"
    # Read the long prompt from the file
    # with open(long_prompt_file, "r", encoding="utf-8") as f:
    #     COAM_LONG_PROMPT = f.read()

    # Get model
    model = config["model"]
    # Get kwargs
    train = kwargs["train"]
    prompt_type = config["prompt_type"]
    task = config["task"]
    seed = config["seed"]
    shots = config["shots"]
    system_prompt = PROMPTS["system"]
    # system_prompt = COAM_LONG_PROMPT
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]
    if "gpt" in model or "gemini" in model or "o1" in model or "o3" in model:
        SCHEMAS = PYDANTIC_SCHEMAS
    else:
        SCHEMAS = TYPED_SCHEMAS

    if "cot" in prompt_type:
        schema = SCHEMAS["MWEsCoT"]
    else:
        schema = SCHEMAS["MWEs"]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = get_few_shot_prompt(
            train, seed=seed, prompt_type=prompt_type, task=task, schemas=SCHEMAS, shots=shots
        )
        # Add in position 1 (after the system prompt)
        messages.insert(1, few_shot_prompt)

    prompt = ChatPromptTemplate.from_messages(messages)

   

    return prompt, schema


def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs
    """
    user_inputs = [{"sentence": (row["sentence"])} for _, row in data.iterrows()]
    return user_inputs


def self_consistency(
    predicted_mwe: list[list[str]], min_occurrence_ratio: float = 0.5
) -> dict[str, float]:
    """
    Calculate the most common mwes from the list of predicted mwes.
    :param predicted_mwe: list of list of predicted mwes
    :param min_occurrence_ratio: minimum ratio of occurrences to consider an mwe valid
    :return: a dictionary with mwes as keys and their occurrence ratio as values
    """
    sc_runs = len(predicted_mwe)
    cleaned_predictions = clean_predictions(predicted_mwe, shortest_version=False)

    # If only one run, return the mwe as is
    if sc_runs == 1:
        if not cleaned_predictions[0]:
            return {}
        else:
            return {mwe: 1.0 for mwe in cleaned_predictions[0]}

    # Flatten the list of lists of cleaned predictions
    flatten_predictions = [pred for sublist in cleaned_predictions for pred in sublist]
    # Count occurrences of each expression
    expression_counts = Counter(flatten_predictions)

    # Calculate the occurrence ratio for each mwe
    expressions_with_ratio = {
        expression: count / sc_runs for expression, count in expression_counts.items()
    }

    # Filter expressions_with_ratio
    expressions_with_ratio = {
        expression: ratio
        for expression, ratio in expressions_with_ratio.items()
        if ratio >= min_occurrence_ratio
    }

    return expressions_with_ratio


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
                if "mwes" in resp["parsed"]:
                    _mwe = resp["parsed"]["mwes"]
                    # Make sure _mwe is a list of strings - use eval
                    if isinstance(_mwe, str):
                        _mwe = eval(_mwe)
                    runs_mwe.append(_mwe)
                elif resp["parsed"] == []:
                    runs_mwe.append([])
                else:
                    # If the response is not in the expected format, skip it
                    continue
                at_least_one_resp_with_no_hallucination = True
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
COAM_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}

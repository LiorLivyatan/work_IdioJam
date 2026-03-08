"""
Utils for the Multi-word Expression Identification (MWEI) task - PARSEME 1.3 dataset.
"""

###############################################################################
# Imports
import os
import json
import pandas as pd
from typing import Callable
from pydantic import BaseModel
import random
from collections import Counter

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.utils import FEW_SHOT_PROMPT_TEMPLATE, MERGE_COLUMNS, clean_predictions, make_examples
from src.pydantic_schemas import PYDANTIC_SCHEMAS
from src.typed_schemas import TYPED_SCHEMAS

###############################################################################

###############################################################################
# Constants

TASK_COLUMNS = MERGE_COLUMNS.copy()

LANGUAGES = [
    "arabic",
    "basque",
    "bulgarian",
    "chinese",
    "croatian",
    "czech",
    "english",
    "farsi",
    "french",
    "german",
    "greek",
    "hebrew",
    "hindi",
    "hungarian",
    "irish",
    "italian",
    "lithuanian",
    "maltese",
    "polish",
    "portuguese",
    "romanian",
    "serbian",
    "slovene",
    "spanish",
    "swedish",
    "turkish",
]

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(parent_dir, "data/parseme")

CUPT_COLUMNS = [
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
    "PARSEME:MWE",
]

###############################################################################


###############################################################################
# Prompts

# type 3
PROMPTS = {
    "system": """You are a helpful system to identify verbal multiple-word expressions (VMWEs).
Identify all the VMWEs in the given sentence, and output their surface forms exactly as they appear.

Definition: VMWEs are sequences of words (continuous or discontinuous) with the following compulsory properties:
They show some degree of orthographic, morphological, syntactic, or semantic idiosyncrasy with respect to what is considered a language's general grammar rules. 
The most salient property of VMWEs is semantic non-compositionality meaning; it is often impossible to deduce the meaning of the whole unit from the meanings of its parts and its syntactic structure.
The three types of VMWEs you should identify are:

(1) LVC.full (Light Verb Construction - Full) - 
A VMWE in which the verb is semantically light — it contributes meaning only through tense, aspect, mood, person, or number — and the noun is predicative, denoting an event or state.
The subject of the verb corresponds to a semantic argument of the noun.
(2) LVC.cause (Light Verb Construction - Causative)
A VMWE in which the verb is causative, meaning it introduces an external cause or source of the event/state described by the noun.
The subject of the verb is not an argument of the noun, but rather the cause behind the noun's event/state.
(3) VID (Verbal Idiom)
A VMWE with at least two lexicalized components, including a head verb and at least one dependent (e.g., noun, prepositional phrase, clause, etc.).
VIDs often carry idiomatic meaning, and the components cannot be freely substituted.
The dependents can be of various syntactic types (subject, object, PP, clause, etc.), and the meaning is typically non-compositional.


Additional instructions:
- You are given one sentence in {language}, you are an expert of this language.
- Be cautious: Only identify VMWEs that clearly match one of the three categories above.
- When listing VMWEs, use exactly the original surface form as it appears in the sentence.
- Only answer in JSON.
""",
    "user": "Sentence: {sentence}\n",
}

###############################################################################


###############################################################################
# Functions

def _extract_mwes(tokens: list[str], tags: list[str]) -> dict[str, list[tuple[str, str]]]:
    """
    Extracts MWEs from tokens and tags.
    :param tokens: list of tokens
    :param tags: list of tags
    :return: dictionary with surface forms and tokenized forms along with their types
    """
    expr_map = {}

    for i, tag in enumerate(tags):
        if tag == "*":
            continue

        for part in tag.split(";"):
            if ":" in part:
                idx, expr_type = part.split(":")
                if idx not in expr_map:
                    expr_map[idx] = {"type": expr_type, "tokens": set()}
                expr_map[idx]["tokens"].add(i)
            elif part.isdigit():
                idx = part
                if idx in expr_map:
                    expr_map[idx]["tokens"].add(i)

    surface_list, surface_tokens_list = [], []

    for expr in expr_map.values():
        expr_type = expr["type"]
        if not expr["tokens"]:
            continue
        indices = sorted(expr["tokens"])

        surface_tokens = [tokens[i] for i in indices]
        surface_form = " ".join(surface_tokens)

        surface_list.append((expr_type, surface_form))
        surface_tokens_list.append((expr_type, surface_tokens))

    return {
        "surface": surface_list,
        "surface_tokens": surface_tokens_list,
    }


def _get_data(data_dir: str, lang: str, data_file: str = '') -> pd.DataFrame:
    """
    Read data from a directory and a specified language.
    :param data_dir: data directory
    :param lang: language
    :param data_file: file name - override data_dir and lang
    :return: data
    """
    new_surface_mapping_file = None

    # Check if data_file is not provided
    if data_file == '':
        data_dir = os.path.join(data_dir, lang)

        # Get the new surface mapping file
        new_surface_mapping_file = os.path.join(data_dir, "new_surface_mapping.json")

        # Find the cupt file in the directory
        data_file = [f for f in os.listdir(data_dir) if f.endswith(".cupt")][0]

        data_file = os.path.join(data_dir, data_file)

    all_data = []
    current_sentence = []
    current_meta = {"source_sent_id": None, "text": None}
    sentence_id = 0

    # Read the file line by line
    with open(data_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("# source_sent_id"):
                    current_meta["source_sent_id"] = line.split("=", 1)[-1].strip()
                elif line.startswith("# text"):
                    current_meta["text"] = line.split("=", 1)[-1].strip()
            elif line == "":
                if current_sentence:
                    for row in current_sentence:
                        row["sentence_id"] = sentence_id
                        row["source_sent_id"] = current_meta.get("source_sent_id")
                        row["text"] = current_meta.get("text")
                        all_data.append(row)
                    current_sentence = []
                    sentence_id += 1
                    current_meta = {"source_sent_id": None, "text": None}
            else:
                parts = line.split("\t")
                if len(parts) == len(CUPT_COLUMNS):
                    token_data = dict(zip(CUPT_COLUMNS, parts))
                    current_sentence.append(token_data)

    words_df = pd.DataFrame(all_data)

    sentences_df = (
        words_df.groupby("sentence_id")
        .agg(
            {
                "source_sent_id": "first",
                "text": "first",
                # Parse all FORM to a list
                "FORM": lambda x: list(x),
                "PARSEME:MWE": lambda x: list(x),
            }
        )
        .reset_index()
    )

    # Change FORM to tokens - column name
    sentences_df.rename(columns={"FORM": "tokens", "PARSEME:MWE": "tags"}, inplace=True)
    # Add language column
    sentences_df["language"] = lang

    sentences_df[["surface", "surface_tokens"]] = sentences_df.apply(
        lambda row: _extract_mwes(row["tokens"], row["tags"]), axis=1, result_type="expand"
    )


    # Init empty column
    sentences_df["fixed_surface"] = [None for _ in range(len(sentences_df))]

    # Check if the new surface mapping file exists
    if new_surface_mapping_file and os.path.exists(new_surface_mapping_file):
        # Read the new surface mapping file
        with open(new_surface_mapping_file, "r", encoding="utf-8-sig") as f:
            new_surface_mapping = json.load(f)

        # Update the surface forms in the DataFrame
        for sentence_id, new_surface in new_surface_mapping.items():
            # Convert each list into a tuple
            new_surface_tuples = [tuple(item) for item in new_surface]

            row_idx = sentences_df[
                sentences_df["sentence_id"] == int(sentence_id)
            ].index
            if len(row_idx) == 1:
                sentences_df.at[row_idx[0], "fixed_surface"] = new_surface_tuples
    return sentences_df


def get_data(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for the given task and prompt type.
    :param kwargs: keyword arguments
    :return: data
    """
    lang = kwargs.get("lang", None)

    # Get data
    test_dir = os.path.join(DATA_DIR, "testset")
    train_dir = os.path.join(DATA_DIR, "trainset")

    train = _get_data(train_dir, lang=lang)
    test = _get_data(test_dir, lang=lang)

    return train, test


def get_few_shot_prompt(
    train: pd.DataFrame, seed, prompt_type: str, schemas, task: str, shots: int = 6, surface=True
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param train: training data
    :param seed: seed for reproducibility
    :param prompt_type: prompt type
    :param schemas: schemas
    :param task: task name
    :param shots: number of few-shot examples
    :param surface: if True, use surface form, else use span form
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility
    # Get few-shot examples for the given language
    if "full" in prompt_type:
        raise NotImplementedError(
            "Full prompt type is not implemented yet. Please use few-shot or zero-shot prompt types."
        )
    else:
        shots = int(shots / 2)  # Get half of the shots for each type
        # Get 3 examples with some wme and 3 examples without mwe
        # mwes is the gold labels
        # Select examples with MWE (non-empty mwes list)
        shots_with_mwe = train[train["surface"].apply(len) > 0].sample(
            shots, random_state=seed
        )
        shots_without_mwe = train[train["surface"].apply(len) == 0].sample(
            shots, random_state=seed
        )

        shots_without_mwe["mwes_final"] = [[] for _ in range(len(shots_without_mwe))]

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
    # Get model
    model = config["model"]
    # Get kwargs
    train = kwargs["train"]
    task = config["task"]

    prompt_type = config["prompt_type"]
    seed = config["seed"]
    shots = config["shots"]

    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    if "gpt" in model or "gemini" in model or "o1" in model or "o3" in model:
        SCHEMAS = PYDANTIC_SCHEMAS
    else:
        SCHEMAS = TYPED_SCHEMAS

    if "cot" in prompt_type:
        schema = SCHEMAS["VMWECoT"]
    else:
        schema = SCHEMAS["VMWE"]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = get_few_shot_prompt(
            train, seed=seed, prompt_type=prompt_type, schemas=SCHEMAS, task=task, shots=shots
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
    user_inputs = [
        {"sentence": row["text"], "language": row["language"]}
        for _, row in data.iterrows()
    ]
    return user_inputs


def self_consistency(
    sentence: str, predicted_mwe: list[list[str]], min_occurrence_ratio: float = 0.5
) -> dict[str, float]:
    """
    Calculate the most common mwes from the list of predicted mwes.
    :param sentence: the sentence to be processed
    :param predicted_mwe: list of list of predicted mwes
    :param min_occurrence_ratio: minimum ratio of occurrences to consider an mwe valid
    :return: a dictionary with mwes as keys and their occurrence ratio as values
    """
    sc_runs = len(predicted_mwe)

    cleaned_predictions = clean_predictions(predicted_mwe)

    # If only one run, return the mwes as is
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
    for res in results:
        sentence = res["text"]
        # Collect PIE from all responses (Self-Consistency)
        runs_mwe = []
        at_least_one_resp_with_no_hallucination = False
        for resp in res["responses"]:
            try:
                if "parsed" in resp:
                    if "vmwes" in resp["parsed"]:
                        _mwes = resp["parsed"]["vmwes"]
                        # Make sure _mwes is a list of strings - use eval
                        if isinstance(_mwes, str):
                            _mwes = eval(_mwes)
                        runs_mwe.append(_mwes)
                    elif resp["parsed"] == []:
                        runs_mwe.append([])
                    else:
                        # If the response is not in the expected format, skip it
                        continue
                else:
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
            best_mwe_with_ratio = self_consistency(sentence, runs_mwe)
            best_mwe = list(best_mwe_with_ratio.keys())
            best_ratios = list(best_mwe_with_ratio.values())
            predicted_mwe.append(best_mwe)
            ratios.append(best_ratios)
            hallucinated.append(False)

    # Add mwes to test data and their ratios
    test["predicted_mwe"] = predicted_mwe
    test["ratios"] = ratios
    test["hallucinated"] = hallucinated
    test['gold'] = test['surface_tokens']

    metrics = {}
    # Collect all labels and predicted tags
    log_cm_report = {}

    kwargs = {
        "gold_col": "fixed_surface",
        "pred_col": "predicted_mwe",
        "tokenized": False,
        "parseme": True,
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
PARSEME_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}

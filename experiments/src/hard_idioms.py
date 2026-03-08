"""
Utils for the Idiom identification task.
"""

###############################################################################
# Imports
import os
import re
import pandas as pd
import random
from typing import Callable
from pydantic import BaseModel
from collections import Counter
import ast


from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.utils import FEW_SHOT_PROMPT_TEMPLATE, MERGE_COLUMNS, read_tsv, clean_predictions, make_examples
from src.pydantic_schemas import PYDANTIC_SCHEMAS
from src.typed_schemas import TYPED_SCHEMAS

###############################################################################

###############################################################################
# Constants
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
LABELS = list(LABEL2ID.keys())

LANGUAGES = [
    "german"
]

TASK_COLUMNS = MERGE_COLUMNS.copy()
for lang in LANGUAGES:
    TASK_COLUMNS.extend(
        [
            f"{lang}_acc",
            f"{lang}_precision",
            f"{lang}_recall",
            f"{lang}_f1",
            f"{lang}_hallucinations",
        ]
    )

# Get parent parent_dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(parent_dir, "data/hard_idioms/production_data/english/hard_idioms_data_english.json")

###############################################################################


###############################################################################
# Prompts


PROMPTS = {
"system": """You are a professional linguist specializing in figurative language and your task is to analyse sentences that may contain an idiom, also known as an idiomatic expression. 
This is a definition of idiom: 'A phrase, expression, or group of words that has a meaning different from the individual meanings of the words themselves, and employed to convey ideas in a non-literal or metaphorical manner'.
Mark idioms only when their usage in the context is idiomatic/figurative and let literal meanings remain unmarked.
You are given one sentence in {language}, you are an expert of this language.
If detected, write the idioms exactly as they are in the sentence, without any changes. Only answer in JSON.
    """,
    "user": "Sentence:{variant_sentence}\n",
}

###############################################################################


###############################################################################
# Functions

def read_bio_tsv(file_path: str) -> pd.DataFrame:
    """
    Reads a BIO-formatted TSV file and returns a Pandas DataFrame.
    Each sentence is treated as a separate sample with a unique ID.
    """
    data = []
    tokens = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # Empty line indicates a new sentence
                if tokens:
                    data.append(([w for w, _ in tokens], [t for _, t in tokens]))
                    tokens = []
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                word, tag = parts
                tokens.append((word, tag))

    # Add the last sentence if not already added
    if tokens:
        data.append([w for w, _ in tokens], [t for _, t in tokens])

    df = pd.DataFrame(data, columns=["tokens", "tags"])

    # Add a column with tag ids
    df["tag_ids"] = df["tags"].apply(lambda x: [LABEL2ID[t] for t in x])

    # Add a column with a list of the MWEs in each sentence
    def extract_idioms(row):
        tags = row["tags"]
        tokens = row["tokens"]
        idioms = []

        while "B-IDIOM" in tags:
            start = tags.index("B-IDIOM")
            # Read until the next "O" tag
            end = start + 1
            while end < len(tags) and tags[end] != "O":
                end += 1
            idioms.append("".join(tokens[start:end]).strip())
            tags = tags[end:]
            tokens = tokens[end:]

        return idioms

    df["true_idioms"] = df.apply(extract_idioms, axis=1)

    # Add a column with the full sentence
    df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

    # Make this column the first one
    df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]

    return df

def get_data(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for the given task and prompt type.
    :param kwargs: keyword arguments
    :return: data
    """
    lang = kwargs.get("lang", "english")  # Get lang from kwargs, default to "english"
    custom_data_path = kwargs.get("data_path", None)  # Support custom data path for correction runs

    # Support custom data path for correction runs
    if custom_data_path:
        test = pd.read_json(custom_data_path)
    else:
        test = pd.read_json(DATA_DIR)   # works when the top-level JSON is a list of dicts

    test["language"] = lang  # Add language column based on config
    return test


def get_few_shot_prompt(
    train: pd.DataFrame, lang: str, seed, prompt_type: str, schemas, task: str, shots: int = 6
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param train: training data
    :param lang: language
    :param seed: seed for reproducibility
    :param prompt_type: prompt type
    :param schemas: schemas
    :param task: task
    :param shots: number of few-shot examples
    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility
    if "full" in prompt_type:
        examples = FEW_SHOT_EXAMPLES[lang]["cot"]
    else:
        shots = int(shots / 2)
        if "pairs" in prompt_type:
            pairs_file_path = os.path.join(DATA_DIR, "id10m_idiom_literal_pairs.tsv")

            # Read pairs file
            pairs_df = read_tsv(pairs_file_path)
            pairs_df = pairs_df[pairs_df["language"] == lang]

            # Sample shots from the pairs
            pairs_df = pairs_df.sample(shots, random_state=seed)

            # Build the examples
            examples = []

            for _, row in pairs_df.iterrows():
                idiomatic_eample = {
                    "input": f"{row['idiomatic']}",
                    "output": f"idioms: [{(row['idiom'])}]",
                }
                literal_example = {
                    "input": f"sentence: {row['literal']}",
                    "output": "idioms: []",
                }
                examples.append(idiomatic_eample)
                examples.append(literal_example)
        else:
            # Get few-shot examples for the given language
            lang_train = train
            # Get 3 examples with some idioms and 3 examples without idioms
            shots_with_idioms = lang_train[
                lang_train["true_idioms"].apply(len) > 0
            ].sample(shots, random_state=seed)
            shots_without_idioms = lang_train[
                lang_train["true_idioms"].apply(len) == 0
            ].sample(shots, random_state=seed)
            # Combine the examples
            few_shot_examples = pd.concat(
                [shots_with_idioms, shots_without_idioms], ignore_index=True
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

    lang = config["lang"]
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

    if "gen" in prompt_type:
        schema = SCHEMAS["IdiomsCoTGen"]
    elif "best" in prompt_type:
        schema = SCHEMAS["IdiomsCoTBest"]
    elif "correction" in prompt_type:
        schema = SCHEMAS["IdiomsCoTCorrection"]
    elif "synonym" in prompt_type:
        schema = SCHEMAS["IdiomsCoTSynonym"]
    elif "cot" in prompt_type:
        schema = SCHEMAS["IdiomsCoT"]
    else:
        schema = SCHEMAS["Idioms"]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        # Only load train data if needed for few-shot
        train_path = os.path.join(parent_dir, f"data/id10m/trainset/{lang}.tsv")
        train = read_bio_tsv(train_path)
        few_shot_prompt = get_few_shot_prompt(
            train,lang=lang ,seed=seed, prompt_type=prompt_type, task=task, schemas=SCHEMAS, shots=shots
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
    # Language column should already exist from get_data, but provide fallback
    if "language" not in data.columns:
        data["language"] = "english"
    user_inputs = [
        {"variant_sentence": "".join(row["final_variant"]), "language": row["language"]}
        for _, row in data.iterrows()
    ]
    return user_inputs


def idioms_list_to_IOB(
    idioms: list[str], variant_sentence: list[str], hallucinated: bool
) -> list[str]:
    """
    Convert a list of idioms to IOB-formatted tags.
    If hallucinated is True, return all "O" tags.
    """
    if hallucinated:
        return ["O"] * len(variant_sentence)

    # Preprocess: strip spaces
    stripped_sentence = [token.strip() for token in variant_sentence]

    # Preprocess: split punctuation like commas and periods
    split_tokens = []
    split_map = []  # Maps split_tokens index back to original sentence index
    for idx, token in enumerate(stripped_sentence):
        # Split punctuation attached to words
        parts = re.findall(r"\w+|[^\w\s]", token, re.UNICODE)
        split_tokens.extend(parts)
        split_map.extend([idx] * len(parts))

    # Lowercase for matching
    split_tokens_lower = [tok.lower() for tok in split_tokens]

    tags = ["O"] * len(variant_sentence)

    try:
        for idiom in sorted(idioms, key=lambda x: -len(x)):  # Shorter idioms first
            # Tokenize idiom: words and punctuation separately
            idiom_tokens = re.findall(r"\w+|[^\w\s]", idiom.lower(), re.UNICODE)

            for i in range(len(split_tokens_lower) - len(idiom_tokens) + 1):
                if split_tokens_lower[i : i + len(idiom_tokens)] == idiom_tokens:
                    # Assign IOB tags according to split_map
                    orig_indices = [
                        split_map[j] for j in range(i, i + len(idiom_tokens))
                    ]
                    first = True
                    for orig_idx in orig_indices:
                        if tags[orig_idx] == "O":  # Don't overwrite if already tagged
                            tags[orig_idx] = "B-IDIOM" if first else "I-IDIOM"
                            first = False
                    break  # Stop after first match
    except Exception as e:
        print(
            f"Error in idioms_list_to_IOB: {e}, idioms: {idioms}, sentence: {variant_sentence}"
        )
        raise ValueError("Error in idioms_list_to_IOB: ", e)

    return tags


def self_consistency(
    predicted_idioms: list[list[str]],
    sc_runs: int = 1,
    min_occurrence_ratio: float = 0.5,
) -> dict[str, float]:
    """
    Calculate the most common idioms from the list of predicted idioms.
    :param predicted_idioms: list of list of predicted idioms
    :param sc_runs: number of self-consistency runs
    :param min_occurrence_ratio: minimum ratio of occurrences to consider an idiom valid
    :return: a dictionary with idioms as keys and their occurrence ratio as values
    """
    # Manually cut SC runs
    predicted_idioms = predicted_idioms[:sc_runs]
    sc_runs = len(predicted_idioms)

    # Flatten the list of lists into a single list
    # all_predictions = [idiom for sublist in predicted_idioms for idiom in sublist]

    cleaned_predictions = clean_predictions(predicted_idioms, shortest_version=False)

    # If only one run, return the idioms as is
    if sc_runs == 1:
        if not cleaned_predictions[0]:
            return {}
        else:
            return {idiom: 1.0 for idiom in cleaned_predictions[0]}

    # Flatten the list of lists of cleaned predictions
    flatten_predictions = [pred for sublist in cleaned_predictions for pred in sublist]
    # Count occurrences of each expression
    expression_counts = Counter(flatten_predictions)

    # Calculate the occurrence ratio for each idiom
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
    sc_runs = kwargs["sc_runs"]
    predicted_idioms = []
    ratios = []
    hallucinated = []
    hallucination_nr = 0

    # Calculate score
    for res in results:
        # Collect PIE from all responses (Self-Consistency)
        runs_idioms = []
        at_least_one_resp_with_no_hallucination = False
        for resp in res["responses"]:
            try:
                if "idioms" in resp["parsed"]:
                    _idioms = resp["parsed"]["idioms"]
                    # Make sure _idioms is a list of strings
                    if isinstance(_idioms, str):
                        # Try to eval if it looks like a list representation
                        if _idioms.strip().startswith('['):
                            try:
                                _idioms = eval(_idioms)
                            except:
                                # If eval fails, split by comma
                                _idioms = [i.strip() for i in _idioms.split(',') if i.strip()]
                        else:
                            # Plain string - split by comma or wrap in list
                            if ',' in _idioms:
                                _idioms = [i.strip() for i in _idioms.split(',') if i.strip()]
                            else:
                                _idioms = [_idioms.strip()] if _idioms.strip() else []
                    runs_idioms.append(_idioms)
                elif resp["parsed"] == []:
                    runs_idioms.append([])
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
            predicted_idioms.append([])
            ratios.append([])
            hallucinated.append(True)
        else:
            # Apply self-consistency to get the most common idioms
            best_idioms_with_ratio = self_consistency(runs_idioms, sc_runs)
            best_idioms = list(best_idioms_with_ratio.keys())
            best_ratios = list(best_idioms_with_ratio.values())
            predicted_idioms.append(best_idioms)
            ratios.append(best_ratios)
            hallucinated.append(False)

    # Add idioms to test data and their ratios
    test["predicted_idioms"] = predicted_idioms
    test["ratios"] = ratios
    test["hallucinated"] = hallucinated

    # Convert idioms to BIO-formatted tags
    test["predicted_tags"] = test.apply(
        lambda x: idioms_list_to_IOB(
            x["predicted_idioms"], x["tokens"], hallucinated=x["hallucinated"]
        ),
        axis=1,
    )

    # Add language column if missing
    if "language" not in test.columns:
        test["language"] = kwargs.get("lang", "english")

    metrics = {}
    # Collect all labels and predicted tags
    log_cm_report = {}
    for lang in test["language"].unique():
        # Get test data for the current language
        lang_test = test[test["language"] == lang]

        lang_labels = [tag for tags in lang_test["tags"] for tag in tags]
        lang_predicted_tags = [
            tag for tags in lang_test["predicted_tags"] for tag in tags
        ]

        # Calculate metrics for the current language
        lang_metrics, conf_matrix, report = calc_metrics(
            lang_labels, lang_predicted_tags, labels=LABELS
        )
        metrics[lang] = lang_metrics
        log_cm_report[lang] = {"conf_matrix": conf_matrix, "report": report}

        # Add the hallucination number to the metrics
        metrics[lang]["hallucinations"] = int(lang_test["hallucinated"].sum())

    # Create a DataFrame with the metrics for CSV based on the res_columns
    csv_metrics = pd.DataFrame(columns=TASK_COLUMNS)
    for lang in LANGUAGES:
        if lang in metrics:
            csv_metrics.loc[0, f"{lang}_acc"] = metrics[lang]["accuracy"]
            csv_metrics.loc[0, f"{lang}_precision"] = metrics[lang]["precision"]
            csv_metrics.loc[0, f"{lang}_recall"] = metrics[lang]["recall"]
            csv_metrics.loc[0, f"{lang}_f1"] = metrics[lang]["f1"]
            csv_metrics.loc[0, f"{lang}_hallucinations"] = metrics[lang][
                "hallucinations"
            ]
    return metrics, test, csv_metrics, log_cm_report


###############################################################################

###############################################################################
# Export
HARD_IDIOMS_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}

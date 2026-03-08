"""
Utils for the Idiom identification task.
"""

###############################################################################
# Imports
import os
import re
import pandas as pd
from typing import Callable
import random
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('punkt_tab')


from src.utils import MERGE_COLUMNS
from src.id10m_utils import get_prompt_schema, get_user_inputs, self_consistency



###############################################################################

###############################################################################
# Constants
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
LABELS = list(LABEL2ID.keys())

LANGUAGES = ["english"]
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
DATA_DIR = os.path.join(parent_dir, "data/magpie")

###############################################################################


###############################################################################
# Functions
def create_iob_tags(row):
    tags = ["O"] * len(row["tokens"])
    
    if row["label"] == "literal":
        return tags
    
    idiom_tokens = word_tokenize(row["true_idioms"][0])
    sentence_tokens = row["tokens"]
    
    # Search for the idiom in the sentence tokens directly
    for i in range(len(sentence_tokens) - len(idiom_tokens) + 1):
        if sentence_tokens[i:i+len(idiom_tokens)] == idiom_tokens:
            for j in range(len(idiom_tokens)):
                tags[i + j] = "B-IDIOM" if j == 0 else "I-IDIOM"
            break
    
    return tags

# Find cases where the true idiom exist more than once in the sentence
def find_multiple_idioms(row):
    idiom = row["true_idioms"]
    if not idiom:
        return False
    sentence = row["sentence"]
    
    # Count occurrences of the idiom in the sentence
    count = len(re.findall(r"\b" + re.escape(idiom[0]) + r"\b", sentence))
    
    return count > 1


def get_data(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for the given task and prompt type.
    :param kwargs: keyword arguments
    :return: data
    """
    task = kwargs["task"]

    train_file = os.path.join(DATA_DIR, "MAGPIE_train_processed.jsonl")

    if "mini" in task:
        test_file = os.path.join(DATA_DIR, "MAGPIE_test_processed_mini.jsonl")
    else:
        test_file = os.path.join(DATA_DIR, "MAGPIE_test_processed.jsonl")

    train = pd.read_json(train_file, lines=True)
    test = pd.read_json(test_file, lines=True)

    return train, test  


def idioms_list_to_IOB(
    idioms: list[str], sentence_tokens: list[str], hallucinated: bool
) -> list[str]:
    """
    Convert a list of idioms to IOB-formatted tags using NLTK word_tokenize for idioms.
    Works with pre-tokenized sentences.
    If hallucinated is True, return all "O" tags.
    
    Args:
        idioms: List of idiom strings to search for in the sentence
        sentence_tokens: The pre-tokenized sentence as a list of tokens
        hallucinated: If True, ignore idioms and return all "O" tags
        
    Returns:
        List of IOB tags corresponding to each token in sentence_tokens
    """
    # Return all "O" tags if hallucinated
    if hallucinated:
        return ["O"] * len(sentence_tokens)
    
    # Initialize all tags as "O"
    tags = ["O"] * len(sentence_tokens)
    
    try:
        # Process idioms in order of length (longest first to avoid nested matches)
        for idiom in sorted(idioms, key=len, reverse=True):
            # Tokenize idiom with NLTK for consistency with other code
            idiom_tokens = word_tokenize(idiom.lower())
            
            # Convert sentence tokens to lowercase for case-insensitive matching
            sentence_tokens_lower = [token.lower() for token in sentence_tokens]
            
            # Search for idiom in sentence
            for i in range(len(sentence_tokens) - len(idiom_tokens) + 1):
                # Compare token sequences
                if sentence_tokens_lower[i:i+len(idiom_tokens)] == idiom_tokens:
                    # Found a match, assign IOB tags
                    for j in range(len(idiom_tokens)):
                        # Only tag if not already tagged (avoid overlapping idioms)
                        if tags[i+j] == "O":
                            tags[i+j] = "B-IDIOM" if j == 0 else "I-IDIOM"
                    break  # Stop after first match for this idiom
                    
    except Exception as e:
        print(f"Error in idioms_list_to_IOB_nltk: {e}, idioms: {idioms}, sentence_tokens: {sentence_tokens}")
        raise ValueError("Error in idioms_list_to_IOB_nltk: ", e)
    
    return tags
    


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
                    # Make sure _idioms is a list of strings - use eval
                    if isinstance(_idioms, str):
                        _idioms = eval(_idioms)
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
MAGPIE_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}

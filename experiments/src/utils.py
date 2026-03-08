"""
Helper functions.
"""

####################################################################################################
# Imports
import os
import logging
from typing import Union, List, Set, Tuple, Dict, FrozenSet, get_args, get_origin, get_type_hints
from pydantic import BaseModel, Field
import json
import re
import pandas as pd
import ast
import itertools
from collections import namedtuple, Counter
import smtplib
from email.mime.text import MIMEText
import yaml
from typing_extensions import Annotated, TypedDict


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.metrics import classification_report
from scipy.stats import pearsonr, spearmanr

from langchain_core.prompts import ChatPromptTemplate

from src.bmc_munkres.munkres import Munkres


####################################################################################################


####################################################################################################
# Constants

FEW_SHOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

MERGE_COLUMNS = ["model", "prompt_type", "seed", "sc_runs", "temperature", "shots"]

####################################################################################################


####################################################################################################
# Functions


def set_keys(keys: dict):
    """
    Set API keys as environment variables.
    :param keys: dictionary with keys
    """
    for key, value in keys.items():
        os.environ[key] = value


def get_logger(name):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
    )
    logger = logging.getLogger(name)
    return logger


def read_tsv(file_path):
    data = pd.read_csv(file_path, sep="\t", quoting=3)
    return data


def is_typeddict(cls):
    return (
        isinstance(cls, type)
        and hasattr(cls, '__annotations__')
        and hasattr(cls, '__total__')  # Only TypedDicts have this
    )

def schema_to_dict_template(schema_class, task: str) -> dict:
    template = {}

    # Handle Pydantic models
    if isinstance(schema_class, type) and issubclass(schema_class, BaseModel):
        for field_name, field in schema_class.model_fields.items():
            field_type = field.annotation

            if get_origin(field_type) == list:
                template[field_name] = []
            elif field_type == str:
                if field_name == "sentence":
                    template[field_name] = ""
                elif field_name in {"idioms", "potential_idioms", "mwes", "vmwes"}:
                    template[field_name] = []
                elif field_name == "explanation":
                    if task in {"id10m", "magpie", "magpie_mini", "hard_id10m"}:
                        template[field_name] = "Let's explain this sentence and the potential idioms in it..."
                    elif task in {"coam", "parseme"}:
                        template[field_name] = "Let's explain the MWEs in the sentence and if they match the definition..."
                else:
                    template[field_name] = ""
            else:
                template[field_name] = None

    # Handle TypedDicts
    elif is_typeddict(schema_class):
        type_hints = get_type_hints(schema_class, include_extras=True)
        for field_name, field_type in type_hints.items():
            if get_origin(field_type) == Annotated:
                field_type = get_args(field_type)[0]

            if get_origin(field_type) == list:
                template[field_name] = []
            elif field_type == str:
                if field_name == "sentence":
                    template[field_name] = ""
                elif field_name in {"idioms", "potential_idioms", "mwes", "vmwes"}:
                    template[field_name] = []
                elif field_name == "explanation":
                    if task in {"id10m", "magpie", "magpie_mini", "hard_id10m"}:
                        template[field_name] = "Let's explain this sentence and the potential idioms in it..."
                    elif task in {"coam", "parseme"}:
                        template[field_name] = "Let's explain the MWEs in the sentence and if they match the definition..."
                else:
                    template[field_name] = ""
            else:
                template[field_name] = None

    else:
        raise TypeError(f"Unsupported schema type: {schema_class}")

    return template

def format_output_from_row(schema_class, row, task: str) -> dict:
    template = schema_to_dict_template(schema_class, task)

    if "sentence" in template:
        template["sentence"] = row["sentence"]

    if "idioms" in template:
        template["idioms"] = row.get("true_idioms", [])

    if "mwes" in template:
        template["mwes"] = row.get("mwes_final", [])

    if "vmwes" in template:
        template["vmwes"] = row.get("mwes_final", [])

    if "potential_idioms" in template:
        template["potential_idioms"] = row.get("true_idioms", [])

    if "figurative_examples" in template:
        # Add only if idioms are present
        if row.get("true_idioms"):
            template["figurative_examples"] = ["Example 1", "Example 2", "Example 3"]
        else:
            template["figurative_examples"] = []

    if "literal_examples" in template:
        # Add only if idioms are present
        if row.get("true_idioms"):
            template["literal_examples"] = ["Example 1", "Example 2", "Example 3"]
        else:
            template["literal_examples"] = []

    if "explanation" in template:
        template["explanation"] = "Let's analyze the expressions in the context of the sentence and explain why they match the definition..."

    return template


def make_examples(few_shot_examples, schemas, schema_type: str, task: str) -> list:
    SCHEMA_MAP = {
        "few_shot_cot": schemas["IdiomsCoT"],
        "few_shot_cot_best": schemas["IdiomsCoTBest"],
        "few_shot_cot_gen": schemas["IdiomsCoTGen"],
        "few_shot_cot_correction": schemas["IdiomsCoTCorrection"],
        "mwes": schemas["MWEs"],
        "mwes_cot": schemas["MWEsCoT"],
        "vmwes": schemas["VMWEs"],
        "vmwes_cot": schemas["VMWEsCoT"],
    }

    schema_class = SCHEMA_MAP.get(schema_type, schemas["Idioms"])

    examples = [
        {
            "input": f"sentence: {row['sentence']}",
            "output": format_output_from_row(schema_class, row, task=task),
        }
        for _, row in few_shot_examples.iterrows()
    ]
    return examples

def clean_predictions(
    all_predictions: list[list[str]], shortest_version: bool = True
) -> list[list[str]]:
    """
    Clean predictions by deduplicating lower-upper-case variants
    and unifying included spans (return the shortest one).
    :param all_predictions: list of lists of predictions
    :param shortest_version: if True, return the shortest version of each mwe
    :return: list of lists of cleaned predictions
    """
    # Remove quotes and double quotes wrapping the mwe
    all_predictions_cleaned = []
    flatten_predictions = []
    for preds in all_predictions:
        preds_cleaned = []
        for pred in preds:
            cleaner_pred = pred.strip('"').strip("'")
            if cleaner_pred:
                # Lowercase the prediction to ignore case
                preds_cleaned.append(cleaner_pred.lower())
                flatten_predictions.append(cleaner_pred.lower())
        all_predictions_cleaned.append(preds_cleaned)

    # Get shortest version of each mwe
    base_predictions = []
    for pred in sorted(list(set(flatten_predictions)), key=len):
        # Check if some base mwe is a substring of the current mwe
        for base_exp in base_predictions:
            if base_exp in pred:
                # If yes, break the loop and do not add the current expression
                break
        else:
            # If not, add the current expression to the list of base expressions
            base_predictions.append(pred)

    # Return a list of base mwes based on their original detection
    cleaned_predictions = []
    for preds in all_predictions_cleaned:
        cleaned_predictions_run = []
        for mwe in preds:
            if shortest_version:
                # Match the base mwe with the original mwe
                for base_exp in base_predictions:
                    if base_exp in mwe and base_exp not in cleaned_predictions_run:
                        cleaned_predictions_run.append(base_exp)
                        break
            else:
                # If not, just add the original mwe to the list of cleaned predictions
                if mwe not in cleaned_predictions_run:
                    cleaned_predictions_run.append(mwe)
        cleaned_predictions.append(cleaned_predictions_run)

    return cleaned_predictions


def parse_json_manually(output: str) -> dict[str, str]:
    """
    Parse reasoning output from the model.
    :param output: output from the model
    :return: dictionary with the answer
    """
    id = output.find("</think>")
    if id != -1:
        # Cut only after the reasoning part
        output = output[id + 8 :]

    # remove markdown code fencing
    clean_json = re.sub(r"^```json\s*|\s*```$", "", output.strip(), flags=re.DOTALL)

    # now parse
    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError:
        # If the output is not a valid JSON, return an empty dictionary
        data = {}

    return data


def parse_response(response: dict, structured: bool = True) -> dict:
    """
    Parse the response from the model
    """
    if not response:
        return {}
    if not structured:
        # If the response is not structured, just return it as is
        return dict(response)

    parsed_response = {}
    parsed_response["raw"] = dict(response["raw"])
    # remove additional_kwargs from the response
    if "additional_kwargs" in parsed_response["raw"]:
        del parsed_response["raw"]["additional_kwargs"]
    parsed_response["parsed"] = {}

    if "parsed" in response and response["parsed"]:
        # If already a a dictionary, just return it
        if isinstance(response["parsed"], dict):
            parsed_response["parsed"] = response["parsed"]
        else:
            # It is our schemas, convert to dict
            try:
                parsed_response["parsed"] = dict(response["parsed"])
            except Exception as e:
                print(
                        f"Error parsing response: {e}, response[parsed]: {response['parsed']}"
                )
                parsed_response["parsed"] = {}
    else:
        # Try to parse the response manually
        try:
            parsed_response["parsed"] = parse_json_manually(response["raw"].content)
        except Exception as e:
            print(
                f"Error parsing response: {e}, parsed_response[parsed]:{parsed_response['parsed']}"
            )
            parsed_response["parsed"] = {}

    return parsed_response


def calc_metrics_cont(y_true, y_pred, **kwargs) -> dict:
    """
    Calculate metrics for regression
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: dictionary with metrics
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearsonr(y_true, y_pred)[0],
        "spearman": spearmanr(y_true, y_pred)[0],
    }

    return metrics, None


def calc_metrics_classification(y_true, y_pred, labels) -> dict:
    """
    Calculate metrics for classification
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to DataFrame for readability
    confusion_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    report = classification_report(y_true, y_pred, target_names=labels)
    return metrics, confusion_df, report



# ------------------------------------------------------------#
# ------------ MWE-based and Token-based ----------------------
# ------------------------------------------------------------#


def clean_and_parse(val: Union[str, list]) -> list:
    """
    function to clean and parse the MWE surface forms
    :param val: MWE list - string or list of strings
    :return: list of MWE tuples (mwe, start, end)
    """
    if isinstance(val, str):
        val = val.strip('"')  # removes surrounding quotes safely
        try:
            return ast.literal_eval(val)
        except:
            return []
    elif isinstance(val, list):
        return val
    return []


def normalize_split(text: str) -> list[str]:
    """
    Normalize the text by lowercasing, replacing dashes with spaces,
    and collapsing multiple spaces. 
    :param text: input text
    :return: list of normalized tokens - split by space
    """
    text = text.lower()
    text = text.replace("-", " ")  # replace dashes with space
    text = text.replace("–", " ")  # replace dashes with space
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip().split(" ")

def keep_only_tag(mwes: list[tuple[str, list[str]]], tags: list[str]) -> list:
    """
    function to keep only the tags in the MWE list
    :param mwes: MWE list
    :param tags: list of tags to keep
    :return: A list MWEs with the tags in the list
    """
    return [mwe for mwe in mwes if mwe[0] in tags]


def seperate_to_tags(test: pd.DataFrame, parseme=False) -> tuple[dict, set]:
    """
    Creates a df for each tag in the test set
    :param test: dataframe with the test set
    :param parseme: if True, computes the trio of VID, LVC.full, LVC.cause
    :return: dictionary with the dataframes for each tag and a set with the tags
    """
    # all the the tags in the test set
    test["gold"] = test["gold"].apply(clean_and_parse)

    tags_sets = set()
    for _, row in test.iterrows():
        gold_mwes = row["gold"] or []
        # Use only first elements (the tag)
        tags = set(mwe[0] for mwe in gold_mwes)
        tags_sets.update(tags)

    tag_to_df = {}
    for tag in tags_sets:
        tag_df = test.copy()
        tag_df["gold"] = tag_df["gold"].apply(lambda x: keep_only_tag(x, [tag]))
        tag_to_df[tag] = tag_df

    # Define the special tag group
    if parseme:
        special_tags = ["VID", "LVC.full", "LVC.cause"]
        combo_name = "_".join(special_tags)
        tag_to_df[combo_name] = test.copy()
        tag_to_df[combo_name]["gold"] = (
            tag_to_df[combo_name]["gold"]
            .apply(lambda x: keep_only_tag(x, special_tags))
            .copy()
        )
        tags_sets.add(combo_name)

    return tag_to_df, tags_sets


def mwe_wordbag(mwe: list[str]) -> Counter:
    """
    Normalize tokens for bag-of-words token comparison.
    """
    return Counter(tok.lower().strip() for tok in mwe)

def mwe_wordset(mwe: list[str]) -> str:
    """
    Normalize tokens into a string for MWE-level comparison.
    """
    return " ".join(tok.lower().strip() for tok in mwe)


# def _calc_metrics_mwe(
#     df: pd.DataFrame) -> dict[str, dict[str, float]]:
#     """
#     Calculate MWE-based and token-based metrics for the given DataFrame.
#     :param df: DataFrame containing the gold and predicted MWE columns
#     :return: Dictionary containing precision, recall, and F1 scores for both MWE-based and token-based metrics
#     """
#     gold_total_mwes = 0
#     pred_total_mwes = 0
#     correct_mwes = 0

#     gold_total_tokens = 0
#     pred_total_tokens = 0
#     correct_tokens = 0

#     for _, row in df.iterrows():
#         gold_mwes = row["gold"] or []
#         pred_mwes = row["predicted_mwe"] or []

#         # Use only second elements (the mwe)
#         gold_mwes = [mwe[1] for mwe in gold_mwes]

#         # Normalize to strings for exact MWE matching
#         gold_sets = set(mwe_wordset(m) for m in gold_mwes)
#         pred_sets = set(mwe_wordset(m) for m in pred_mwes)

#         correct_mwes += len(gold_sets & pred_sets)
#         gold_total_mwes += len(gold_sets)
#         pred_total_mwes += len(pred_sets)

#         # Token-based comparison (fuzzy)
#         gold_bags = [mwe_wordbag(m) for m in gold_mwes]
#         pred_bags = [mwe_wordbag(m) for m in pred_mwes]

#         gold_total_tokens += sum(sum(bag.values()) for bag in gold_bags)
#         pred_total_tokens += sum(sum(bag.values()) for bag in pred_bags)

#         for g_bag in gold_bags:
#             for p_bag in pred_bags:
#                 correct_tokens += sum((g_bag & p_bag).values())

#     # MWE-based scores
#     p_mwe = correct_mwes / pred_total_mwes if pred_total_mwes else 0
#     r_mwe = correct_mwes / gold_total_mwes if gold_total_mwes else 0
#     f1_mwe = 2 * p_mwe * r_mwe / (p_mwe + r_mwe) if (p_mwe + r_mwe) else 0

#     # Token-based scores
#     p_tok = correct_tokens / pred_total_tokens if pred_total_tokens else 0
#     r_tok = correct_tokens / gold_total_tokens if gold_total_tokens else 0
#     f1_tok = 2 * p_tok * r_tok / (p_tok + r_tok) if (p_tok + r_tok) else 0

#     return {
#         "mwe_based": {"P": p_mwe, "R": r_mwe, "F1": f1_mwe},
#         "token_based": {"P": p_tok, "R": r_tok, "F1": f1_tok},
#     }



def _calc_metrics_mwe(df: pd.DataFrame, debug: bool = False) -> dict[str, dict[str, float]]:
    debug_mwe_file = "debug_mwe.txt"
    debug_token_file = "debug_token.txt"
    gold_total_mwes = 0
    pred_total_mwes = 0
    correct_mwes = 0

    gold_total_tokens = 0
    pred_total_tokens = 0
    correct_tokens = 0

    for _, row in df.iterrows():
        gold_mwes = row["gold"] or []
        pred_mwes = row["predicted_mwe"] or []

        # Normalize to lowercase, stripped tokens
        gold_sets = [mwe_wordset(mwe[1]) for mwe in gold_mwes]
        pred_sets = [mwe_wordset(mwe) for mwe in pred_mwes]
        # Log if gold is not empty
        # if debug and gold_sets:
        #     with open(debug_mwe_file, "a", encoding="utf-8") as f:
        #         # Write the sentences
        #         f.write(f"Sentence: {row['text']}\n")
        #         f.write(f"Gold: {gold_sets}\n")
        #         f.write(f"Pred: {pred_sets}\n\n")

        # One-to-one exact match
        matched_gold = set()
        matched_pred = set()
        for i, g in enumerate(gold_sets):
            for j, p in enumerate(pred_sets):
                if g == p and j not in matched_pred and i not in matched_gold:
                    matched_gold.add(i)
                    matched_pred.add(j)
                    correct_mwes += 1
                    break

        gold_total_mwes += len(gold_sets)
        pred_total_mwes += len(pred_sets)

        # --- Token-based optimal matching ---
        gold_bags = [mwe_wordbag(mwe[1]) for mwe in gold_mwes]
        pred_bags = [mwe_wordbag(mwe) for mwe in pred_mwes]
        # Log
        # if debug and gold_bags:
        #     with open(debug_token_file, "a", encoding="utf-8") as f:
        #         f.write(f"Gold: {[dict(bag) for bag in gold_bags]}\n")
        #         f.write(f"Pred: {[dict(bag) for bag in pred_bags]}\n")

        gold_total_tokens += sum(sum(bag.values()) for bag in gold_bags)
        pred_total_tokens += sum(sum(bag.values()) for bag in pred_bags)

        if gold_bags and pred_bags:
            cost_matrix = [[-sum((g & p).values()) for p in pred_bags] for g in gold_bags]
            try:
                indexes = Munkres().compute(cost_matrix)
                for i, j in indexes:
                    if i < len(gold_bags) and j < len(pred_bags):
                        # Log the matched pairs
                        # with open(debug_token_file, "a", encoding="utf-8") as f:
                        #     f.write(f"Matched: {list((gold_bags[i] & pred_bags[j]).keys())}\n\n")
                        correct_tokens += sum((gold_bags[i] & pred_bags[j]).values())
            except Exception:
                pass

    # Metrics
    p_mwe = correct_mwes / pred_total_mwes if pred_total_mwes else 0
    r_mwe = correct_mwes / gold_total_mwes if gold_total_mwes else 0
    f1_mwe = 2 * p_mwe * r_mwe / (p_mwe + r_mwe) if (p_mwe + r_mwe) else 0

    p_tok = correct_tokens / pred_total_tokens if pred_total_tokens else 0
    r_tok = correct_tokens / gold_total_tokens if gold_total_tokens else 0
    f1_tok = 2 * p_tok * r_tok / (p_tok + r_tok) if (p_tok + r_tok) else 0

    return {
        "mwe_based": {"P": p_mwe, "R": r_mwe, "F1": f1_mwe},
        "token_based": {"P": p_tok, "R": r_tok, "F1": f1_tok}
    }


def calc_metrics_mwe(test: pd.DataFrame, **kwargs) -> dict[str, float]:
    """
    function to calculate metrics for each split of the data: tags and full
    Expects gold_col and pred_col, tokenized and parseme as kwargs
    :param test: dataframe with the test set
    :param kwargs: dictionary with the gold_col, pred_col, tokenized and parseme
    :return: dictionary with the metrics for each tag and the full data
    """

    gold_col = kwargs.get("gold_col", "gold")
    pred_col = kwargs.get("pred_col", "predicted_mwe")
    tokenized = kwargs.get("tokenized", False)
    text_col = "text" if "text" in test.columns else "sentence"
    # Create a new df with the gold and predicted columns
    test = test[[gold_col, pred_col, text_col]].copy()
    test = test.rename(columns={gold_col: "gold", pred_col: "predicted_mwe", text_col: "text"})
    
    # Clean the gold and predicted columns
    test["gold"] = test["gold"].apply(clean_and_parse)
    test["predicted_mwe"] = test["predicted_mwe"].apply(clean_and_parse)

    # If not tokenized, normalize the text and then split
    if not tokenized:
        test["gold"] = test["gold"].apply(
            lambda x: [(mwe[0], normalize_split(mwe[1])) for mwe in x]
        )
        test["predicted_mwe"] = test["predicted_mwe"].apply(
            lambda x: [normalize_split(mwe) for mwe in x]
        )

    if "parseme" in kwargs:
        parseme = kwargs["parseme"]
    else:
        parseme = False

    # geting the data split according to the tags
    tag_split_df, tags_sets = seperate_to_tags(test, parseme)
    metrics = {}

    for tag in tags_sets:
        metrics[tag] = _calc_metrics_mwe(tag_split_df[tag])

    # calculate the metrics for the full data
    metrics["full"] = _calc_metrics_mwe(test, debug=False)

    return metrics





def send_email(
    config_path: str = "src/mail_config.yaml",
    app_password: str = '',
    sender_email: str = '',
    receiver_email: str = '',
    subject: str = "Python Script Finished",
    body: str = "Hey, your Python script just finished running."
) -> int:
    """
    Send an email notification using SMTP.
    If config_path is provided, it will load the sender's email and app password from the config file.
    If no receiver email is provided, it defaults to the sender's email.
    :param sender_email: Sender's email address
    :param receiver_email: Receiver's email address
    :param app_password: App password for the sender's email account
    :param subject: Subject of the email
    :param body: Body of the email
    :return: 1 if successful, 0 if failed
    """
    if app_password == '' or sender_email == '':
        # Make sure the config file exists
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                sender_email = config.get("sender_email")
                app_password = config.get("app_password")
        except FileNotFoundError:
            print(f"❌ Config file '{config_path}' not found.")
            return 0

    if receiver_email == '':
        receiver_email = sender_email
    
    # Create the email message
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ Email sent successfully.")
        return 1
    except Exception as e:
        print("❌ Failed to send email:", e)
        return 0

####################################################################################################

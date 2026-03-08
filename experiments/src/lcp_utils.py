"""
Utils for the Language Complexity Prediction (LCP) task.
"""

###############################################################################
# Imports
import os
import numpy as np
import pandas as pd
import random
from typing import Callable
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from src.utils import read_tsv

###############################################################################

###############################################################################
# Constants
LABEL2SCORE = {
    "very easy": 0.0,
    "easy": 0.25,
    "neutral": 0.5,
    "difficult": 0.75,
    "very difficult": 1.0,
}

###############################################################################


###############################################################################
# Prompts

FEW_SHOT_EXAMPLES = [
    {
        "sentence": "Because your loving kindness is better than life, my lips shall praise you.",
        "word": "loving kindness",
        "proof": "Loving kindness is very easy because both words are common and convey positive emotions. Their combined meaning remains clear and intuitive for beginners.",
        "complex": "very easy",
    },
    {
        "sentence": "They also allow for easy compensation for the thousands of accidents involving vehicles from more than one Member State.",
        "word": "easy compensation",
        "proof": "Easy compensation is easy because both words are common, and the phrase clearly means simple reimbursement, making it understandable for learners.",
        "complex": "easy",
    },
    {
        "sentence": "by Crescenzio Rivellini, on behalf of the Committee on Budgetary Control, on discharge in respect of the implementation of the European Union general budget for the financial year 2009, Section IV - Court of Justice (SEC(2010)0963 - C7-0214/2010 -;",
        "word": "financial year",
        "proof": "Financial year is neutral because it has a specific meaning in accounting and business, making it familiar to intermediate learners but challenging for beginners.",
        "complex": "neutral",
    },
    {
        "sentence": "These findings demonstrate very distinct activities of one transcriptional regulator at different developmental steps within a committed post-mitotic neuronal lineage.",
        "word": "neuronal lineage",
        "proof": "Neuronal lineage is difficult because both words are specialized scientific terms, making the phrase complex for learners without a biology background.",
        "complex": "difficult",
    },
    {
        "sentence": "Additionally, two patients with deletions apparently encompassing the FOG2 locus have died from multiple congenital anomalies including CDH [38-40].",
        "word": "congenital anomalies",
        "proof": "Congenital anomalies iis very difficult because both words are specialized medical terms, making the phrase challenging for learners without a medical background.",
        "complex": "very difficult",
    },
]

PROMPTS = {
    "lcp_multi": {
        "system": """You are a helpful, honest, and respectful assistant for identifying the word complexity for beginner English learners.
        You are given one sentence in English and a phrase from that sentence.
        Your task is to evaluate the complexity of the word. Answer with one of the following:
        very easy, easy, neutral, difficult, very difficult. Be concise. 
        """,
        "user": "What is the difficulty of {token} from {sentence}?",
    },
    "lcp_single": {
        "zero_shot": "Enter the string: ",
    },
}
###############################################################################


###############################################################################
# Schemas


class ComplexitySchema(BaseModel):
    """The complexity class of a token: very easy, easy, neutral, difficult, very difficult"""

    complex: str = Field(
        description="either very easy, easy, neutral, difficult, or very difficult"
    )


class ComplexityCoTSchema(BaseModel):
    """The complexity class of a token: very easy, easy, neutral, difficult, very difficult"""

    sentence: str = Field(description="the sentence you were provided")
    word: str = Field(description="the word or words you have to analyze")
    proof: str = Field(description="explain your response in maximum 50 words")
    complex: str = Field(
        description="either very easy, easy, neutral, difficult, or very difficult"
    )


SCHEMAS = {
    "lcp_multi": {
        "zero_shot": ComplexitySchema,
        "few_shot": ComplexitySchema,
        "zero_shot_cot": ComplexityCoTSchema,
        "few_shot_cot": ComplexityCoTSchema,
    },
    "lcp_single": {
        "zero_shot": ComplexityCoTSchema,
    },
}

###############################################################################


###############################################################################
# Functions


def get_data(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get data for the given task and prompt type.
    :param kwargs: keyword arguments
    :return: data
    """
    # Get parent parent_dir
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get data
    if task == "lcp_multi":
        train_path = os.path.join(parent_dir, "data/lcp/train/lcp_multi_train.tsv")
        test_path = os.path.join(parent_dir, "data/lcp/test/lcp_multi_test.tsv")
    elif task == "lcp_single":
        train_path = os.path.join(parent_dir, "data/lcp/train/lcp_single_train.tsv")
        test_path = os.path.join(parent_dir, "data/lcp/test/lcp_single_test.tsv")
    else:
        raise ValueError(f"Task {task} is not supported")

    train = read_tsv(train_path)
    test = read_tsv(test_path)

    return train, test


def _add_few_shots(system_prompt: str, cot: bool = False) -> str:
    """
    Add shuffled few-shot examples to the system prompt.

    :param system_prompt: The original system prompt
    :param cot: Whether the task is CoT
    :return: The system prompt with few-shot examples appended
    """
    # Shuffle
    examples = random.sample(FEW_SHOT_EXAMPLES, len(FEW_SHOT_EXAMPLES))

    # Remove proof from examples if not CoT
    if not cot:
        examples = [{k: v for k, v in ex.items() if k != "proof"} for ex in examples]

    # Format examples
    few_shot_text = "\n\nHere are some examples:\n"
    for ex in examples:
        # Add all key-value pairs
        for k, v in ex.items():
            few_shot_text += f"{k}: {v}\n"
        few_shot_text += "\n"

    return system_prompt + few_shot_text.strip()  # Append and return


def get_prompt_schema(task: str, prompt_type: str) -> tuple[ChatPromptTemplate, dict]:
    """
    Get the prompt for the given task and prompt type.
    :param task: task name
    :param prompt_type: prompt type
    :return: prompt and schema
    """
    system_prompt = PROMPTS[task]["system"]
    user_prompt = PROMPTS[task]["user"]

    if "few_shot" in prompt_type:
        system_prompt = _add_few_shots(system_prompt, "cot" in prompt_type)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )
    schema = SCHEMAS[task][prompt_type]
    return prompt, schema


def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs
    """
    user_inputs = [
        {"sentence": row["sentence"], "token": row["token"]}
        for _, row in data.iterrows()
    ]
    return user_inputs


def get_metrics(
    results: list[dict], test: pd.DataFrame, calc_metrics: Callable
) -> tuple[dict, pd.DataFrame]:
    """
    Calculate metrics for the given task after post-processing the results.
    :param results: results
    :param test: test data
    :return: metrics and test data after post-processing
    """
    model_scores = []
    model_stds = []
    # Calculate score
    for res in results:
        runs_scores = [LABEL2SCORE[resp["complex"]] for resp in res["responses"]]
        res["model_score"] = np.mean(runs_scores)
        res["model_std"] = np.std(runs_scores)
        model_scores.append(res["model_score"])
        model_stds.append(res["model_std"])

    # Add model scores to test data
    test["model_score"] = model_scores
    test["model_std"] = model_stds

    # Calculate metrics
    metrics = calc_metrics(test["complexity"], test["model_score"])
    return metrics, test


###############################################################################

###############################################################################
# Export
LCP_UTILS = {
    "get_data": get_data,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "get_metrics": get_metrics,
}

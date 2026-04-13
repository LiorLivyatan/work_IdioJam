"""
Running an experiment on a dataset
"""

import os
import yaml
import wandb
import json
import argparse
from argparse import Namespace
import pandas as pd
from transformers import set_seed
from datetime import datetime

from src.utils import (
    MERGE_COLUMNS,
    get_logger,
    set_keys,
    calc_metrics_cont,
    calc_metrics_classification,
    parse_response,
    calc_metrics_mwe,
    send_email
)
from src.models import get_model

from src.lcp_utils import LCP_UTILS
from src.id10m_utils import ID10M_UTILS
from src.coam_utils import COAM_UTILS
# from src.coam_tsv_utils import COAM_TSV_UTILS
from src.parseme_utils import PARSEME_UTILS
from src.parseme_vid_utils import PARSEME_VID_UTILS
from src.magpie_utils import MAGPIE_UTILS
from src.hard_idioms import HARD_IDIOMS_UTILS


# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="config.yaml",
    help="Path to the config file",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--lang", type=str, default=None, help="Language")
parser.add_argument(
    "--sc_runs",
    type=int,
    default=None,
    help="Number of self-consistency runs",
)
parser.add_argument(
    "--responses_dir",
    type=str,
    default=None,
    help="Directory with responses",
)
parser.add_argument(
    "--resume_dir",
    type=str,
    default=None,
    help="Resume a partial SC run from this directory (skips completed batches)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="Custom data file path (for correction runs)",
)

####################################################################################################
# Functions


def get_task_utils(task: str):
    """
    Get utils for the given task.
    :param task: task name
    :return: utils
    """
    if task == "lcp_multi" or task == "lcp_single":
        utils = LCP_UTILS
        calc_metrics = calc_metrics_cont
    elif task == "id10m":
        utils = ID10M_UTILS
        calc_metrics = calc_metrics_classification
    elif task == "coam":
        utils = COAM_UTILS
        calc_metrics = calc_metrics_mwe
    # elif task == "coam_tsv":
    #     utils = COAM_TSV_UTILS
    #     calc_metrics = calc_metrics_mwe
    elif task == "parseme":
        utils = PARSEME_UTILS
        calc_metrics = calc_metrics_mwe
    elif task == "parseme_vid":
        utils = PARSEME_VID_UTILS
        calc_metrics = calc_metrics_mwe
    elif task == "magpie_mini":
        utils = MAGPIE_UTILS
        calc_metrics = calc_metrics_classification
    elif task == "hard_idioms":
        utils = HARD_IDIOMS_UTILS
        calc_metrics = calc_metrics_classification
    elif task == "magpie":
        # Let the user know its big and we dont support it yet
        raise NotImplementedError(
            "Magpie is too big. Please use magpie_mini instead."
        )
    else:
        raise ValueError(f"Task {task} is not supported")
    return Namespace(
        get_data=utils["get_data"],
        get_prompt_schema=utils["get_prompt_schema"],
        get_user_inputs=utils["get_user_inputs"],
        calc_metrics=calc_metrics,
        process_responses=utils["process_responses"],
    )


def create_run_directory(base_exp_dir: str) -> str:
    """
    Create a new run directory with auto-increment numbering.
    
    Args:
        base_exp_dir: Base experiment directory path
        
    Returns:
        Full path to the new run directory
    """
    # Ensure base experiment directory exists
    os.makedirs(base_exp_dir, exist_ok=True)
    
    # Find existing run directories
    existing_runs = []
    if os.path.exists(base_exp_dir):
        for item in os.listdir(base_exp_dir):
            item_path = os.path.join(base_exp_dir, item)
            if os.path.isdir(item_path) and item.startswith('run_'):
                try:
                    # Extract run number from 'run_XXX' format
                    run_num = int(item.split('_')[1])
                    existing_runs.append(run_num)
                except (IndexError, ValueError):
                    # Skip directories that don't follow run_XXX format
                    continue
    
    # Determine next run number
    if existing_runs:
        next_run = max(existing_runs) + 1
    else:
        next_run = 1
    
    # Create new run directory name
    run_dir_name = f"run_{next_run:03d}"  # Format as run_001, run_002, etc.
    run_dir_path = os.path.join(base_exp_dir, run_dir_name)
    
    # Create the run directory
    os.makedirs(run_dir_path, exist_ok=True)
    
    return run_dir_path


def main():
    # Get logger
    logger = get_logger(__name__)

    # Get CMD args
    cmd_args = parser.parse_args()
    logger.info(f"CMD args: {cmd_args}")

    # Load keys
    with open("keys.yaml", "r") as f:
        keys = yaml.safe_load(f)
    logger.info("Loaded API keys")
    # Set keys
    set_keys(keys)

    # Load config
    config_file = cmd_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config: {config}")

    if "responses_dir" in cmd_args and cmd_args.responses_dir:
        config["responses_dir"] = cmd_args.responses_dir
        logger.info(f"Updated config with responses_dir: {config['responses_dir']}")

    # Check if responses were given
    if config["responses_dir"]:
        responses_dir = config["responses_dir"]
        # Get the original config
        with open(os.path.join(config["responses_dir"], "config.yaml"), "r") as f:
            orig_config = yaml.safe_load(f)
        # Update config with the original config
        config.update(orig_config)
        # Update responses_dir to the original one
        config["responses_dir"] = responses_dir
        logger.info(f"Updated config from {config['responses_dir']}")

    # Update config with CMD args
    for key, value in vars(cmd_args).items():
        # Skip None values
        if value is None:
            continue
        config[key] = value

    # Set seed
    set_seed(config["seed"])

    # Get experiment name
    model_name = config["model"].split("/")[-1]
    exp_name = f"{config['task']}_{model_name}_{config['prompt_type']}_shots_{config['shots']}_sc{config['sc_runs']}_tmp{config['temperature']}_seed{config['seed']}"

    # Get utils
    task_utils = get_task_utils(config["task"])

    # Add language to experiment name
    if config["lang"]:
        exp_name += f"_{config['lang']}"

    # Assert
    if "few" in config["prompt_type"]:
        assert config["shots"] > 0, "Shots must be greater than 0"
        assert config["shots"] % 2 == 0, "Shots must be even"
    if "zero" in config["prompt_type"]:
        assert config["shots"] == 0, "Shots must be 0 for zero-shot"
    if config["sc_runs"] > 1:
        assert (
            config["temperature"] == 0.8
        ), "Temperature must be 0.8 for self-consistency"

    # Load task results
    # Create language-specific subdirectories for tasks that support language specification
    if config["lang"] and config["task"] in ["parseme", "id10m", "hard_idioms"]:
        task_res_dir = os.path.join(
            config["results_dir"], config["task"], config["lang"]
        )
    else:
        task_res_dir = os.path.join(config["results_dir"], config["task"])
    os.makedirs(task_res_dir, exist_ok=True)

    task_res_file = os.path.join(task_res_dir, "full_results.csv")
    if os.path.exists(task_res_file):
        task_res = pd.read_csv(task_res_file)
    else:
        task_res = pd.DataFrame(columns=MERGE_COLUMNS)
        task_res.to_csv(task_res_file, index=False)

    # Handle resume: reuse an existing partial-run directory
    resume_dir = cmd_args.resume_dir
    if resume_dir:
        exp_dir = os.path.abspath(resume_dir)
        if not os.path.isdir(exp_dir):
            raise ValueError(f"--resume_dir does not exist: {exp_dir}")
        run_number = os.path.basename(exp_dir)
        logger.info(f"Resuming run from {exp_dir}")
    else:
        # Create experiment directory with new structure
        # logs/hard_idioms/german/hard_idioms_gpt-4o-mini_zero_shot_shots_0_sc1_tmp0.3_seed42_german/run_001/
        # Include language subdirectory for tasks that support language specification
        if config["lang"] and config["task"] in ["parseme", "id10m", "hard_idioms"]:
            mother_dir = os.path.join(config["logs_dir"], config["task"], config["lang"])  # logs/hard_idioms/german/
        else:
            mother_dir = os.path.join(config["logs_dir"], config["task"])  # logs/hard_idioms/
        base_exp_dir = os.path.join(mother_dir, exp_name)  # logs/hard_idioms/german/hard_idioms_gpt-4o-mini_.../
        exp_dir = create_run_directory(base_exp_dir)  # logs/hard_idioms/german/hard_idioms_gpt-4o-mini_.../run_001/

        logger.info(f"Created experiment directory: {exp_dir}")
        run_number = os.path.basename(exp_dir)
        logger.info(f"This is {run_number} for this experiment configuration")

    # Add experiment metadata
    config["experiment_start_date"] = datetime.now().strftime("%Y-%m-%d")
    config["experiment_start_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["model_checkpoint"] = "pending"  # Will be updated after first response

    # Write config to file
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Initialize W&B
    if not config["debug"]:
        wandb.login()
        wandb.init(
            project=config["task"],
            config=config,
            name=exp_name,
            reinit=True,
        )

    def convert_nan_to_none(obj):
        """Recursively convert NaN values to None for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(item) for item in obj]
        elif isinstance(obj, float) and pd.isna(obj):
            return None
        else:
            return obj

    # Load data
    test = task_utils.get_data(
        task=config["task"],
        lang=config["lang"],
        data_path=config.get("data_path")
    )
    train = None

    # Cut only respective language
    if config["lang"]:
        if "language" in test.columns:
            # train = train[train["language"] == config["lang"]]
            test = test[test["language"] == config["lang"]]

    # if train is not None:
    #     logger.info(f"Loaded train data: {train.shape}")
    logger.info(f"Loaded test data: {test.shape}")

    # Cut data for debugging
    if config["debug"]:
        if "debug_samples" in config and config.get("debug_samples") is not None:
            # Choose those samples from the test set
            test = test.iloc[config["debug_samples"]]
        else:
            test = test[:config["num_samples"]]
        # train = train[:config["num_samples"]] # original
        # if train is not None:
        #     train = train.sample(min(len(test), 100), random_state=config["seed"])
        

    # Check if responses were given
    if config["responses_dir"]:
        responses_file = os.path.join(config["responses_dir"], "responses.json")
        with open(responses_file, "r", encoding="utf-8-sig") as f:
            responses = json.load(f)
        logger.info(f"Loaded responses: {len(responses)}")

    else:
        # Initialize model
        llm = get_model(config["model"], config["temperature"], config["use_rate_limiter"])

        # Get prompt template and schema
        prompt, schema = task_utils.get_prompt_schema(config=config, train=train)

        if schema:
            # Apply structured output if required
            llm = llm.with_structured_output(schema, include_raw=True)
            structured = True
        else:
            structured = False

        # Build chain
        chain = prompt | llm

        # prepare data
        user_inputs = task_utils.get_user_inputs(test)

        # Aggregate results — load partial responses if resuming, else start fresh
        partial_responses_file = os.path.join(exp_dir, "responses.json")
        if resume_dir and os.path.exists(partial_responses_file):
            with open(partial_responses_file, "r", encoding="utf-8-sig") as f:
                responses = json.load(f)
            start_run = min(len(r["responses"]) for r in responses)
            logger.info(f"Resuming from SC batch {start_run} (already have {start_run}/{config['sc_runs']} batches)")
        else:
            responses = []
            for _, row in test.iterrows():
                _data = {key: row[key] for key in test.columns}
                _data["responses"] = []
                responses.append(_data)
            start_run = 0

        # Run with multiple seeds/runs
        for run_index in range(start_run, config["sc_runs"]):
            if config["batched"]:
                logger.info(f"Running batch for run {run_index}")
                try:
                    raw_responses_run = chain.batch(user_inputs)
                    logger.info(
                        f"Generated {len(raw_responses_run)} raw responses for run {run_index}"
                    )
                except Exception as e:
                    logger.error(f"Error during batch run {run_index}: {e}")
                    raw_responses_run = []
                    continue
            else:
                # Open a file to save the problematic inputs
                problematic_inputs_file = os.path.join(
                    exp_dir, f"problematic_inputs_run_{run_index}.txt"
                )
                with open(problematic_inputs_file, "w") as f:
                    f.write("Problematic inputs:\n")
                logger.info(f"Running individual invokes for run {run_index}")

                def invoke_row(i, row):
                    try:
                        return chain.invoke(row)
                    except Exception as e_individual:
                        logger.warning(
                            f"Error for input {i} in run {run_index}: {e_individual}"
                        )
                        logger.warning(f"Problematic input: {row}")
                        # Save the problematic input to the file
                        with open(problematic_inputs_file, "a") as f:
                            f.write(f"Input {i}: {row}\n")
                            f.write(f"Error: {e_individual}\n\n")
                        return None

                # Apply the function to each row in the DataFrame
                raw_responses_run = list(
                    map(
                        lambda args: invoke_row(*args),
                        zip(range(len(user_inputs)), user_inputs),
                    )
                )

            # Capture model checkpoint from first valid response
            if config.get("model_checkpoint") == "pending" and raw_responses_run and run_index == 0:
                for resp in raw_responses_run:
                    if resp:
                        try:
                            # Extract checkpoint from response metadata
                            if hasattr(resp, 'response_metadata'):
                                checkpoint = resp.response_metadata.get("model_name", config["model"])
                            elif isinstance(resp, dict) and "response_metadata" in resp:
                                checkpoint = resp["response_metadata"].get("model_name", config["model"])
                            else:
                                checkpoint = config["model"]

                            config["model_checkpoint"] = checkpoint
                            logger.info(f"Captured model checkpoint: {checkpoint}")

                            # Update config file with checkpoint
                            with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
                                yaml.dump(config, f)
                            break
                        except Exception as e:
                            logger.warning(f"Could not extract model checkpoint: {e}")
                            config["model_checkpoint"] = config["model"]

            # Save responses to results
            for i, resp in enumerate(raw_responses_run):
                try:
                    responses[i]["responses"].append(parse_response(resp, structured))
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
                    responses[i]["responses"].append({})

            # Checkpoint: save after every successful SC batch so we can resume on failure
            _checkpoint_clean = convert_nan_to_none(responses)
            with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
                json.dump(_checkpoint_clean, f, indent=1, ensure_ascii=False)
            logger.info(f"Checkpointed responses after SC batch {run_index + 1}/{config['sc_runs']}")

    responses_clean = convert_nan_to_none(responses)

    with open(os.path.join(exp_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(responses_clean, f, indent=1, ensure_ascii=False)
    logger.info(f"Saved responses to {exp_dir}")

    # Calculate metrics
    try:
        metrics, test, run_res, log_cm_report = task_utils.process_responses(
            responses,
            test,
            task_utils.calc_metrics,
            lang=config["lang"],
            sc_runs=config["sc_runs"],
        )
    except Exception as e:
        raise RuntimeError(f"Error processing responses: {e}")

    logger.info(f"Metrics: {metrics}")

    # Log metrics
    if not config["debug"]:
        wandb.log({"metrics": metrics})

    # Write metrics to file
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrices and classification reports
    if log_cm_report:
        with open(os.path.join(exp_dir, "conf_matrices_reports.txt"), "w") as f:
            for lang in log_cm_report.keys():
                cm = log_cm_report[lang]["conf_matrix"]
                report = log_cm_report[lang]["report"]
                f.write(f"Language: {lang}\n")
                f.write(f"Confusion matrix:\n{cm}\n\n")
                f.write(f"Classification report:\n{report}\n")
                f.write("\n")

    # Save results to file
    test.to_csv(os.path.join(exp_dir, "results.tsv"), index=False, sep="\t")

    # Save results to task results
    for col in MERGE_COLUMNS:
        run_res[col] = config[col]

    if not config["debug"]:
        # Merge and update while preserving structure
        task_res = task_res.set_index(MERGE_COLUMNS)
        run_res = run_res.set_index(MERGE_COLUMNS)

        task_res.update(run_res)  # Update existing rows only

        task_res = task_res.combine_first(run_res)  # Add new rows if they don’t exist

        # Reset index and save
        task_res.reset_index(inplace=True)

        task_res.to_csv(task_res_file, index=False)

    logger.info(f"Saved results to {exp_dir}")

    # Finish W&B
    if not config["debug"]:
        wandb.finish()
        # send_email(
        #     subject=f"Experiment {exp_name} finished",
        #     body=f"Experiment {exp_name} finished. Check the results at {exp_dir}",
        # )


####################################################################################################

####################################################################################################
# Main

if __name__ == "__main__":
    main()
    
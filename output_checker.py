from typing import Dict, List, Optional
import glob
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from pyhailing import RidehailEnv


def get_eval_files(the_dir: str) -> List[str]:
    """Provides a list of all evaluation files in the directory."""
    
    if the_dir is None:
        the_dir = "./"
    
    if the_dir[-1] != '/':
        the_dir = the_dir + "/"
    
    return glob.glob(the_dir + "pyhailing_eval_results*.json")


def load_result_file(eval_file) -> Optional[Dict]:
    """Loads the results data from a submitted file.
    
    If the file cannot be read, returns None.
    
    """

    try:
        with open(eval_file, 'r') as f:
            result = json.load(f)
            return result

    except Exception:
        return None


def get_instance_category(entry) -> Optional[str]:
    """Determines the instance category for which the entry was submitted.
    
    If it does not match the config of any instance category, returns None.
    
    """
    
    instance_categories = RidehailEnv.DIMACS_CONFIGS.ALL_CONFIGS
    
    entry_config = entry["config"]
    keys_to_check = list(entry_config.keys())
    
    try:
        keys_to_check.remove("nickname")
    except:
        return None


    for category_name, category_config in instance_categories.items():
        if all((entry_config[key] == category_config[key] for key in keys_to_check)):
            return category_name

    return None


def get_rwd_upperbound(env) -> float:
    """Provides an upperbound on the reward for the env's current episode."""

    # TODO
    ub = np.inf

    return ub


def check_vehicle_sequences(assignments, env) -> None:
    """Checks that assignments are plausible."""

    # TODO

    return True


def get_entry_performance(entry):
    """Checks the performance of an entry."""
    
    eps_results = entry["episodes"]

    # Make sure the proper number of episodes were performed
    expected_num_eps = RidehailEnv.DIMACS_NUM_EVAL_EPISODES
    
    if len(eps_results) != expected_num_eps:
        logging.warning(
            f"Invalid entry. Expected results for {len(eps_results)} episodes but got results for {expected_num_eps}.")
        return None
    
    # Create an environment to check the solution
    config = entry["config"]
    config["for_evaluation"] = False
    env = RidehailEnv(entry["config"])

    # For each episode...
    for i, eps_result in enumerate(eps_results):
        
        env.reset()
        
        # Make sure the claimed reward does not exceed a computable UB
        rwd_upperbound = get_rwd_upperbound(env)

        if eps_result['final_reward'] > rwd_upperbound:
            logging.warning("Invalid reward achieved.")
            return None

        # Get the sequence for each vehicle
        # get the sequence of reqs for each vehicle. for each, make sure it is time feasible (with an easy check)
        seqs_feasible = check_vehicle_sequences(eps_result['assignments'], env)
        if not seqs_feasible:
            logging.warning("Invalid vehicle-to-request assignments.")
            return None

    # All good. Get the mean episode reward
    mean_reward = (
        sum((eps_result["final_reward"] for eps_result in eps_results))
        / expected_num_eps
    )

    return mean_reward


def get_competition_results(eval_files:List[str]) -> pd.DataFrame:

    # Initialize the results of the competition
    competition_results = []

    for eval_file in eval_files:

        # Initialize a results record for this entry
        entry_results = {
            "filename": eval_file,
            "instance_category": None,
            "mean_reward": None,
        }

        # Load the file
        entry = load_result_file(eval_file)

        if entry is None:
            logging.warning(f"Entry file could not be read: {eval_file}")
            competition_results.append(entry_results)
            continue

        # Determine the entry's instance category
        instance_category = get_instance_category(entry)

        if instance_category is None:
            logging.warning(
                f"Entry file's config does not match the config of any instance category: {eval_file}"
            )
            competition_results.append(entry_results)
            continue
        
        entry_results["instance_category"] = instance_category
        
        # Get the entry's mean episode reward
        mean_reward = get_entry_performance(entry)

        # Done.
        competition_results.append(entry_results)
    
    # Combine all results into df and return it
    return pd.DataFrame(competition_results)


if __name__ == "__main__":

    now = time.strftime("%Y%m%d%H%M%S")

    args = sys.argv

    if len(args) != 2:
        logging.warning("Grabbing eval files from the current directory.")
        eval_files_dir = "./"
    
    else:
        eval_files_dir = args[1]

    eval_files = get_eval_files(eval_files_dir)
    
    eval_results = get_competition_results(eval_files)
    
    eval_results.to_csv(f"competition_results_{now}.csv", index=None)
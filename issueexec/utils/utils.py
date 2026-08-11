"""
Utility functions for data loading, logging, and file operations.
"""

import json
import logging
import os
from collections import defaultdict


def load_jsonl(filepath: str) -> list:
    """
    Load a JSONL file from the given filepath.

    Args:
        filepath: The path to the JSONL file to load.

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_jsonl(data: list, filepath: str) -> None:
    """
    Write data to a JSONL file at the given filepath.

    Args:
        data: A list of dictionaries to write to the JSONL file.
        filepath: The path to the JSONL file to write.
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def load_json(filepath: str) -> dict:
    """
    Load a JSON file from the given filepath.

    Args:
        filepath: The path to the JSON file to load.

    Returns:
        A dictionary representing the JSON data.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def combine_by_instance_id(data: list) -> list:
    """
    Combine data entries by their instance ID.

    Args:
        data: A list of dictionaries with instance IDs and other information.

    Returns:
    A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for key, value in item.items():
            if key != "instance_id":
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{"instance_id": iid}, **details} for iid, details in combined_data.items()
    ]


def setup_logger(log_file: str) -> logging.Logger:
    """
    Set up a logger that writes to a file.

    Args:
        log_file: The path to the log file.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def cleanup_logger(logger: logging.Logger) -> None:
    """
    Clean up a logger by removing and closing all handlers.

    Args:
        logger: The logger instance to clean up.
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def load_existing_instance_ids(output_file: str) -> set:
    """
    Load existing instance IDs from an output file.

    Args:
        output_file: The path to the output file.

    Returns:
        A set of instance IDs that have already been processed.
    """
    instance_ids = set()

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "instance_id" in data:
                        instance_ids.add(data["instance_id"])
                except json.JSONDecodeError:
                    continue

    return instance_ids

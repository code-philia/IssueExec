# issueexec/utils/get_repo_structure/get_repo_structure.py
import argparse
import ast
import json
import os
import subprocess
import uuid

import pandas as pd
from tqdm import tqdm

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clone_repo(repo_name, repo_playground):
    try:

        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{repo_to_top_folder[repo_name]}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{repo_to_top_folder[repo_name]}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_project_structure_from_scratch(
    repo_name, base_commit, instance_id, repo_playground
):
    """
    Build repository structure for a specific instance.
    Note: use per-instance clone directory to avoid collisions in parallel runs.
    """

    # ========== Per-instance clone path strategy ==========
    # Alternative (disabled):
    # simple_repo_name = repo_name.replace('/', '_')
    # repo_dir = os.path.join(repo_playground, f"{simple_repo_name}__{instance_id}")

    # Current strategy:
    repo_dir = os.path.join(repo_playground, instance_id)
    # ===============================================================

    # If repo already exists and commit matches, reuse it directly.
    if os.path.exists(repo_dir):
        try:
            result = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            current_commit = result.stdout.strip()

            if current_commit == base_commit:
                print(f"Reusing existing repository at {repo_dir}")
                structure = create_structure(repo_dir)
                return {
                    "repo": repo_name,
                    "base_commit": base_commit,
                    "structure": structure,
                    "instance_id": instance_id,
                    "repo_path": repo_dir
                }
            else:
                print(f"Wrong commit ({current_commit} vs {base_commit}), re-cloning...")
                subprocess.run(["rm", "-rf", repo_dir], check=True)
        except Exception as e:
            print(f"Error checking existing repo: {e}, re-cloning...")
            if os.path.exists(repo_dir):
                subprocess.run(["rm", "-rf", repo_dir], check=True)

    # Clone repository into target directory.
    # os.makedirs(repo_dir, exist_ok=True)
    # os.makedirs(repo_dir, exist_ok=True)

    try:
        print(f"Cloning repository to {repo_dir}...")
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", repo_dir],
            check=True,
            capture_output=True,  # Keep stdout/stderr for diagnostics
            text=True             # Decode output as text
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e}")
        print(f"STDERR: {e.stderr}")  # Helpful git failure details
        raise

    # Checkout target commit.
    checkout_commit(repo_dir, base_commit)

    # Parse repository structure.
    structure = create_structure(repo_dir)

    # Return both structure and local repo path for later cleanup.

    return {
        "repo": repo_name,
        "base_commit": base_commit,
        "structure": structure,
        "instance_id": instance_id,
        "repo_path": repo_dir  # Keep local path for downstream steps
    }


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure

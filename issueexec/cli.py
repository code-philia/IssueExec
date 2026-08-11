# issueexec/cli.py
import os
from typing import Optional, Dict, List, Any
from functools import partial
from issueexec.localizer import RelatedTestRetriever, BlindSpotAnalyzer, SuppletoryLocalizer, Reranker
import json
import argparse
import concurrent.futures
from multiprocessing import Manager

from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from issueexec.utils.utils import load_existing_instance_ids, load_jsonl, setup_logger
from issueexec.utils.preprocess_data import (
    check_contains_valid_loc,
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)

import warnings
warnings.filterwarnings("ignore")


def basic_processing(issue, args):
    instance_id = issue["instance_id"]

    # Get repository structure
    result = get_repo_structure(
        instance_id, issue["repo"], issue["base_commit"], args.repo_cache_dir
    )

    # Compatible with both old and new version return values
    if isinstance(result, dict) and "structure" in result:
        structure = result["structure"]
        repo_path = result.get("repo_path")
    else:
        structure = result
        repo_path = None

    # Extract problem statement directly from issue
    problem_statement = issue["problem_statement"]

    # Filter
    filter_none_python(structure)

    return structure, problem_statement, repo_path


def save_data_with_write_lock(args, write_lock, data_dict):
    assert "instance_id" in data_dict, "Data must contain 'instance_id' key"
    if write_lock is not None:
        write_lock.acquire()

    try:
        with open(args.output_file, "a") as f:
            f.write(json.dumps(data_dict) + "\n")
    finally:
         if write_lock is not None:
            write_lock.release()


# stage1
def related_tests_retrieval(
    issue,
    args,
    existing_instance_ids=None,
    write_lock=None,
    coverage_graph_path: Optional[str] = None,
    test_functions_path: Optional[str] = None,
    expand_query: bool = True,
    domain_knowledge_path: Optional[str] = None,
    use_online_domain_knowledge: bool = False
):
    """
    Input: Issue description
    Process: Retrieve test cases related to the Issue based on semantic similarity and functional module mapping
    Output: Related Tests set + test coverage path information
    """

    # Set logger
    instance_id = issue["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing issue {instance_id} with related tests retrieval")

    # Skip if already processed
    if existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {issue['instance_id']}")
        return

    logger.info(f"================ related tests retrieval {instance_id} ================")
    structure, problem_statement, repo_path = basic_processing(issue, args)


    # Initialize result variables
    found_tests = []
    additional_artifact_loc_test = None
    test_traj = {}

    related_test_retriever = RelatedTestRetriever(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        coverage_graph_path=coverage_graph_path,
        test_functions_path=test_functions_path,
        expand_query=expand_query,
        domain_knowledge_path=domain_knowledge_path if not use_online_domain_knowledge else None,
        use_online_domain_knowledge=use_online_domain_knowledge,
        repo=issue['repo'] if use_online_domain_knowledge else None,
        base_commit=issue['base_commit'] if use_online_domain_knowledge else None,
        repo_path=repo_path
    )

    # Find relevant test functions
    found_tests, additional_artifact_loc_test, test_traj = related_test_retriever.localize(
            top_n=getattr(args, 'top_n', 5),  # Add new arg for number of tests
            mock=args.mock
    )

    logger.info(f"Found {len(found_tests)} relevant test functions: {found_tests}")


    found_test_names = [test['name'] for test in found_tests]
    # Map to coverage elements
    coverage_elements = related_test_retriever.extract_coverage_elements(
                found_test_names
    )

    logger.info(f"Found {len(coverage_elements)} coverage elements from coverage analysis")

    result = {
                "instance_id": instance_id,
                "found_tests": found_tests,
                "additional_artifact_loc_test": additional_artifact_loc_test,
                "test_traj": test_traj,
                "coverage_elements": coverage_elements
            }

    save_data_with_write_lock(args, write_lock, result)

    if repo_path and os.path.exists(repo_path):
        try:
            import shutil
            logger.info(f"[CLEANUP] Attempting to delete repository: {repo_path}")
            shutil.rmtree(repo_path)
            logger.info(f"[CLEANUP] Successfully deleted repository: {repo_path}")
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete repository {repo_path}: {e}")
            import traceback
            logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")
    elif repo_path:
        logger.warning(f"[CLEANUP] Repository path does not exist: {repo_path}")
    else:
        logger.warning(f"[CLEANUP] No repository path to clean up (repo_path is None)")

# stage2
def blind_spot_analysis(
    issue,
    args,
    start_file_locs,
    existing_instance_ids=None,
    write_lock=None,
    coverage_graph_path: Optional[str] = None,
    domain_knowledge_path: Optional[str] = None
):
    """
    Executed after related_tests_retrieval

    start_file is the saved file from the related_tests_retrieval stage output
    """

    # Set logger
    instance_id = issue["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing issue {instance_id} with related tests retrieval")

    # Skip if already processed
    if existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {issue['instance_id']}")
        return

    logger.info(f"================ blind spot analysis {instance_id} ================")
    structure, problem_statement, repo_path = basic_processing(issue, args)

    # Load data from previous stage
    for locs in start_file_locs:
        if locs['instance_id'] == instance_id:
            found_tests = locs.get("found_tests", [])
            additional_artifact_loc_test = locs.get("additional_artifact_loc_test")
            test_traj = locs.get("test_traj", {})
            coverage_elements = locs.get("coverage_elements", {})

    logger.info("Starting test blind spots analysis...")


    # # Expand coverage elements using coverage graph
    # if coverage_graph_path:
    #     coverage_elements = expand_coverage_with_graph(
    #         instance_id=instance_id,
    #         coverage_elements=coverage_elements,
    #         coverage_graph_path=coverage_graph_path,
    #         logger=logger
    #     )
    #     logger.info(f"Expanded coverage elements using graph")
    logger.info(f"Using coverage_elements from stage1: {len(coverage_elements)} tests")


    blind_spot_analyzer = BlindSpotAnalyzer(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        found_tests,
        coverage_elements,
        domain_knowledge_path=domain_knowledge_path
    )


    # Analyze test blind spots
    (
        blind_spots_analysis,
        additional_artifact_blind_spots,
        blind_spots_traj
    ) = blind_spot_analyzer.analyze_blind_spots(
        mock=args.mock
    )

    logger.info(f"Completed blind spots analysis: {blind_spots_analysis}")


    # Map suspicious locations based on blind spots
    (
        suspicious_locations,
        additional_artifact_suspicious_locations,
        suspicious_locations_traj
    ) = blind_spot_analyzer.localize_suspicious_locations(
        blind_spots_analysis,
        mock=args.mock
    )


    logger.info(f"Found {len(suspicious_locations)} suspicious functions from blind spots analysis")

    result = {
                "instance_id": instance_id,
                "blind_spots_analysis": blind_spots_analysis,
                "additional_artifact_blind_spots": additional_artifact_blind_spots,
                "blind_spots_traj": blind_spots_traj,

                "suspicious_locations": suspicious_locations,
                "additional_artifact_suspicious_locations": additional_artifact_suspicious_locations,
                "suspicious_locations_traj": suspicious_locations_traj
            }

    save_data_with_write_lock(args, write_lock, result)

    if repo_path and os.path.exists(repo_path):
        try:
            import shutil
            logger.info(f"[CLEANUP] Attempting to delete repository: {repo_path}")
            shutil.rmtree(repo_path)
            logger.info(f"[CLEANUP] Successfully deleted repository: {repo_path}")
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete repository {repo_path}: {e}")
            import traceback
            logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")
    elif repo_path:
        logger.warning(f"[CLEANUP] Repository path does not exist: {repo_path}")
    else:
        logger.warning(f"[CLEANUP] No repository path to clean up (repo_path is None)")


def load_hierarchy_info(hierarchy_data_path, instance_id, log):
    log.info(f"Loading hierarchy info for instance {instance_id} from {hierarchy_data_path}")
    with open(os.path.join(hierarchy_data_path, instance_id + ".json"), "r") as f:
        hierarchy_info = json.load(f)

    return hierarchy_info

# stage3
def enhanced_localization(
    issue,
    args,
    start_file_locs,
    existing_instance_ids=None,
    write_lock=None,
    hierarchy_data_path: Optional[str] = None,
    historical_data_path: Optional[str] = None
):
    """
    Executed after blind_spot_analysis

    Input: Issue description + suspicious locations list
    Process: Use hierarchy information and historical fix information to enhance localization accuracy
    Output: Precise suspicious function/class locations

    Args:
        issue: SWE-bench issue data
        args: Command line arguments
        swe_bench_data: SWE-bench dataset
        start_file_locs: Output file content from previous stage (blind_spot_analysis)
        existing_instance_ids: Set of processed instance_ids
        write_lock: Multi-thread write lock
        hierarchy_data_path: Hierarchy information file path
        historical_data_path: Historical fix information file path
    """

    # Set logger
    instance_id = issue["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing issue {instance_id} with enhanced localization")

    # Skip if already processed
    if existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {issue['instance_id']}")
        return

    logger.info(f"================ enhanced localization {instance_id} ================")
    structure, problem_statement, repo_path = basic_processing(issue, args)

    # Load data from previous stage
    suspicious_locations = []
    for locs in start_file_locs:
        if locs['instance_id'] == instance_id:
            suspicious_locations = locs.get("suspicious_locations", [])
            break

    if not suspicious_locations:
        logger.warning(f"No suspicious locations found for instance {instance_id}")
        result = {
            "instance_id": instance_id,
            "enhanced_locations": [],
            "additional_artifact_discriminates": {},
            "discriminate_trajs": {}
        }
        save_data_with_write_lock(args, write_lock, result)
        return

    logger.info(f"Found {len(suspicious_locations)} suspicious locations to analyze")

    # Load hierarchy and historical information
    hierarchy_info = load_hierarchy_info(hierarchy_data_path, instance_id, logger)
    historical_info = None
    # historical_info = _load_historical_info(historical_data_path, instance_id, suspicious_locations, logger)

    logger.info("Starting enhanced localization analysis...")

    # Instantiate EnhancedLocalizer
    enhanced_localizer = EnhancedLocalizer(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        suspicious_locations,
        hierarchy_info,
        historical_info
    )

    # Execute enhanced localization
    (
        enhanced_locations,
        additional_artifact_discriminates,
        discriminate_trajs
    ) = enhanced_localizer.localize(
        top_n=getattr(args, 'top_n', 3),
        mock=args.mock
    )

    logger.info(f"Enhanced localization completed: found {len(enhanced_locations)} high-confidence locations")
    logger.info(f"Enhanced locations: {list(enhanced_locations)}")

    result = {
        "instance_id": instance_id,
        "enhanced_locations": list(enhanced_locations),
        "additional_artifact_discriminates": additional_artifact_discriminates,
        "discriminate_trajs": discriminate_trajs
    }

    save_data_with_write_lock(args, write_lock, result)

    if repo_path and os.path.exists(repo_path):
        try:
            import shutil
            logger.info(f"[CLEANUP] Attempting to delete repository: {repo_path}")
            shutil.rmtree(repo_path)
            logger.info(f"[CLEANUP] Successfully deleted repository: {repo_path}")
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete repository {repo_path}: {e}")
            import traceback
            logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")
    elif repo_path:
        logger.warning(f"[CLEANUP] Repository path does not exist: {repo_path}")
    else:
        logger.warning(f"[CLEANUP] No repository path to clean up (repo_path is None)")



def suppletory_retrieval(
        issue,
        args,
        start_file_locs,
        existing_instance_ids,
        write_lock,
        context_level: str = "file"
):
    """
    Executed after blind_spot_analysis

    Input: Issue description + suspicious locations list
    Process: Extract suspicious files from suspicious locations list (i.e., extract suspicious_file from suspicious_file::suspicious_func) and perform supplementary localization in suspicious files
    Output: Supplementary localized class/function locations

    Args:
        issue: SWE-bench issue data
        args: Command line arguments
        swe_bench_data: SWE-bench dataset
        start_file_locs: Output file content from previous stage (blind_spot_analysis)
        existing_instance_ids: Set of processed instance_ids
    """

    # Set logger
    instance_id = issue["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing issue {instance_id} with enhanced localization")

    # Skip if already processed
    if existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {issue['instance_id']}")
        return

    logger.info(f"================ suppletory retrieval {instance_id} ================")
    structure, problem_statement, repo_path = basic_processing(issue, args)

    # Load data from previous stage
    suspicious_locations = []
    for locs in start_file_locs:
        if locs['instance_id'] == instance_id:
            suspicious_locations = locs.get("suspicious_locations", [])
            break

    if not suspicious_locations:
        logger.warning(f"No suspicious locations found for instance {instance_id}")
        result = {
            "instance_id": instance_id,
            "suppletory_locations": [],
            "additional_artifact_suppletory": {},
            "suppletory_trajs": {}
        }
        save_data_with_write_lock(args, write_lock, result)
        return

    # Extract different granularity contexts based on context_level
    if context_level == "file":
        # File level: extract file paths
        suspicious_contexts = list(set([item.split("::")[0] for item in suspicious_locations]))
        logger.info(f"Found {len(suspicious_locations)} suspicious locations, derived {len(suspicious_contexts)} unique files (file-level context)")
    else:
        # Module level: extract file::ClassName format
        suspicious_contexts = []
        for item in suspicious_locations:
            if "::" in item:
                file_path, identifier = item.split("::", 1)
                # Extract class name (if in ClassName.method_name format, take ClassName)
                if "." in identifier:
                    class_name = identifier.split(".")[0]
                    module_context = f"{file_path}::{class_name}"
                else:
                    # Could be class name or function name, keep as is
                    module_context = item
                suspicious_contexts.append(module_context)
        suspicious_contexts = list(set(suspicious_contexts))
        logger.info(f"Found {len(suspicious_locations)} suspicious locations, derived {len(suspicious_contexts)} unique modules (module-level context)")

    suppletory_localizer = SuppletoryLocalizer(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        suspicious_contexts,
        context_level=context_level
    )

    # Execute supplementary localization
    (
        suppletory_locations,
        additional_artifact_suppletory,
        suppletory_trajs
    ) = suppletory_localizer.localize(
        top_n=getattr(args, 'top_n', 3),
        mock=args.mock
    )

    logger.info(f"Suppletory localization completed: found {len(suppletory_locations)} high-confidence locations")
    logger.info(f"Suppletory locations: {list(suppletory_locations)}")

    result = {
        "instance_id": instance_id,
        "suppletory_locations": list(suppletory_locations),
        "additional_artifact_suppletory": additional_artifact_suppletory,
        "suppletory_trajs": suppletory_trajs
    }

    save_data_with_write_lock(args, write_lock, result)

    if repo_path and os.path.exists(repo_path):
        try:
            import shutil
            logger.info(f"[CLEANUP] Attempting to delete repository: {repo_path}")
            shutil.rmtree(repo_path)
            logger.info(f"[CLEANUP] Successfully deleted repository: {repo_path}")
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete repository {repo_path}: {e}")
            import traceback
            logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")
    elif repo_path:
        logger.warning(f"[CLEANUP] Repository path does not exist: {repo_path}")
    else:
        logger.warning(f"[CLEANUP] No repository path to clean up (repo_path is None)")



def reranking(
        issue,
        args,
        start_file_locs,
        existing_instance_ids,
        write_lock
):
    """
    Rank results, executed after suppletory_retrieval and enhanced_localization
    Input: Issue description + output results from each stage
    Process: Aggregate output results from each stage and perform ranking
    Output: Final localization results

    Args:
        issue: SWE-bench issue data
        args: Command line arguments
        swe_bench_data: SWE-bench dataset
        start_file_locs: Merged output file content
        existing_instance_ids: Set of processed instance_ids
    """

    # Set logger
    instance_id = issue["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing issue {instance_id} with enhanced localization")

    # Skip if already processed
    if existing_instance_ids and instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {issue['instance_id']}")
        return

    logger.info(f"================ reranking {instance_id} ================")
    structure, problem_statement, repo_path = basic_processing(issue, args)

    # Load data from previous stage
    locations = []
    for locs in start_file_locs:
        if locs['instance_id'] == instance_id:
            locations = locs.get("locations", [])
            break

    if not locations:
        logger.warning(f"No locations found for instance {instance_id}")
        result = {
            "instance_id": instance_id,
            "sorted_locations": [],
            "additional_artifact_sorted": {},
            "sorted_trajs": {}
        }
        save_data_with_write_lock(args, write_lock, result)
        return

    logger.info(f"Found {len(locations)} locations to analyze")


    logger.info("Starting reranking analysis...")


    # Pass context_expansion parameter when instantiating Reranker
    # Get configuration from args, default is False for backward compatibility
    context_expansion = getattr(args, 'context_expansion', False)

    # Instantiate Reranker
    reranker = Reranker(
        instance_id,
        structure,
        problem_statement,
        args.model,
        args.backend,
        logger,
        locations,
        context_expansion=context_expansion
    )

    # Execute ranking
    (
        sorted_locations,
        additional_artifact_sorted,
        sorted_trajs
    ) = reranker.localize(
        mock=args.mock
    )

    logger.info(f"Reranking completed: reranking {len(sorted_locations)} locations")
    logger.info(f"Sorted locations: {list(sorted_locations)}")
    result = {
        "instance_id": instance_id,
        "sorted_locations": list(sorted_locations),
        "additional_artifact_sorted": additional_artifact_sorted,
        "sorted_trajs": sorted_trajs
    }

    save_data_with_write_lock(args, write_lock, result)

    if repo_path and os.path.exists(repo_path):
        try:
            import shutil
            logger.info(f"[CLEANUP] Attempting to delete repository: {repo_path}")
            shutil.rmtree(repo_path)
            logger.info(f"[CLEANUP] Successfully deleted repository: {repo_path}")
        except Exception as e:
            logger.error(f"[CLEANUP] Failed to delete repository {repo_path}: {e}")
            import traceback
            logger.error(f"[CLEANUP] Traceback: {traceback.format_exc()}")
    elif repo_path:
        logger.warning(f"[CLEANUP] Repository path does not exist: {repo_path}")
    else:
        logger.warning(f"[CLEANUP] No repository path to clean up (repo_path is None)")


def localize(args):
    try:
        swe_bench_data = load_dataset(args.dataset, split="test")
    except Exception as e:
        # load from disk
        print(f"Loading {args.dataset} from disk")
        swe_bench_data = load_from_disk(args.dataset)

    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    coverage_graph_path = args.coverage_graph_path
    test_functions_path = args.test_functions_path
    hierarchy_data_path = getattr(args, 'hierarchy_data_path', None)
    historical_data_path = getattr(args, 'historical_data_path', None)
    expand_query = getattr(args, 'expand_query', False)
    domain_knowledge_path = getattr(args, 'domain_knowledge_path', None)
    use_online_domain_knowledge = getattr(args, 'use_online_domain_knowledge', False)
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    swe_bench_list = list(swe_bench_data)  # Convert to list to ensure serializability

    seen = set()
    filtered_issues = []
    for issue in swe_bench_list:
        iid = issue['instance_id']
        # Skip already processed and duplicates within dataset
        if iid not in existing_instance_ids and iid not in seen:
            seen.add(iid)
            filtered_issues.append(issue)

    print(f"[Localize] Total issues in dataset: {len(swe_bench_list)}")
    print(f"[Localize] Already processed (skip): {len(existing_instance_ids)}")
    print(f"[Localize] Duplicate in dataset (skip): {len(swe_bench_list) - len(set(i['instance_id'] for i in swe_bench_list))}")
    print(f"[Localize] To process: {len(filtered_issues)}")

    # Select corresponding worker function and parameters based on stage
    if args.stage == "related_tests_retrieval":
        worker_func = partial(
            related_tests_retrieval,
            args=args,
            existing_instance_ids=existing_instance_ids,
            coverage_graph_path=coverage_graph_path,
            test_functions_path=test_functions_path,
            expand_query=expand_query,
            domain_knowledge_path=domain_knowledge_path,
            use_online_domain_knowledge=use_online_domain_knowledge
        )
    elif args.stage == "blind_spot_analysis":
        worker_func = partial(
            blind_spot_analysis,
            args=args,
            start_file_locs=start_file_locs,
            existing_instance_ids=existing_instance_ids,
            coverage_graph_path=coverage_graph_path,
            domain_knowledge_path=domain_knowledge_path
        )
    elif args.stage == "suppletory_retrieval":
        worker_func = partial(
            suppletory_retrieval,
            args=args,
            start_file_locs=start_file_locs,
            existing_instance_ids=existing_instance_ids,
            context_level=getattr(args, 'suppletory_context_level', 'file')
        )
    elif args.stage == "reranking":
        worker_func = partial(
            reranking,
            args=args,
            start_file_locs=start_file_locs,
            existing_instance_ids=existing_instance_ids
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    if args.num_threads == 1:
        for issue in tqdm(filtered_issues, colour="MAGENTA"):
            worker_func(issue, write_lock=None)
    else:
        write_lock = Manager().Lock()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(worker_func, issue, write_lock=write_lock)
                for issue in filtered_issues
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(filtered_issues), colour="MAGENTA"):
                future.result()




def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )

    # Retrieval stage
    parser.add_argument("--stage", choices=["related_tests_retrieval",
                                            "blind_spot_analysis",
                                            # "enhanced_localization",
                                            "suppletory_retrieval",
                                            "reranking"])

    parser.add_argument("--expand_query", default=False, help="Whether to perform query expansion in the localization process.")


    # Model
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-coder",
        choices=[
            "gpt-4o-2024-05-13",
            "deepseek-coder",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="deepseek",
        choices=["openai", "deepseek", "anthropic"],
    )


    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="SWE-bench/SWE-bench_Lite",
        help="Dataset for evaluation (supports HuggingFace dataset name or local path)",
    )
    parser.add_argument(
        "--repo_cache_dir",
        type=str,
        default="/tmp/swe_bench_repos",
        help="Directory for temporary repository checkouts.",
    )



    # Input data
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--keep_old_order", action="store_true")


    # Parallel
    # Actually uses processes (multiprocessing), not threads
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )

    # test coverage (unified to coverage graph format)
    parser.add_argument(
        "--coverage_graph_path",
        default=None,
        help="Test coverage graph with nodes and edges"
    )

    parser.add_argument(
        "--test_functions_path", default = None, help = "list of test functions"
    )

    # Reranking
    parser.add_argument(
        "--context_expansion",
        action="store_true",
        default=True,
        help="Enable code context expansion in reranking stage. "
             "When enabled, extracts and includes full code context for each location. "
             "Default: False (uses only location names, compatible with old behavior)"
    )


    # Suppletory Retrieval
    parser.add_argument(
        "--suppletory_context_level",
        type=str,
        default="file",
        choices=["file", "module"],
        help="Context level for suppletory retrieval: 'file' for entire file, 'module' for class/module level"
    )

    # Domain knowledge Enhancement
    parser.add_argument(
        "--domain_knowledge_path",
        default="",
        help="Path to domain knowledge JSON file containing enriched test representations"
    )

    parser.add_argument(
        "--use_online_domain_knowledge",
        action="store_true",
        default=False,
        help="Enable online domain knowledge collection after BM25 filtering (recommended for faster processing). "
             "When enabled, domain knowledge is collected in real-time for BM25-selected tests only, "
             "significantly reducing processing time compared to offline pre-computation."
    )


    # Other
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--direct_edit_loc", action="store_true")
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )

    parser.add_argument(
        "--hierarchy_data_path",
        default=None,
        help="Path to hierarchy data files"
    )

    parser.add_argument(
        "--historical_data_path", default = None
    )


    args = parser.parse_args()
    args.output_folder = os.path.join(args.output_folder, args.stage)
    args.output_file = os.path.join(args.output_folder, args.output_file)

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # Write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    localize(args)


if __name__ == "__main__":
    main()

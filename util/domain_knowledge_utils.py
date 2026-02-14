#!/usr/bin/env python3
# util/domain_knowledge_utils.py
"""
Online Domain Knowledge collection tool

For a small set of test functions filtered by BM25, this tool collects historical information
and domain knowledge in real time.
"""

import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import threading

# ==================== Reuse the core classes and functions from the original scripts ====================
# Note:
# - collect_historical_info.py and domain_knowledge_enhancement.py
#   should be placed under utils/ directory (same level as this file)

try:
    # Dynamically import collect_historical_info.py
    import sys
    import importlib.util

    hist_info_path = os.path.join(os.path.dirname(__file__), 'collect_historical_info.py')
    if not os.path.exists(hist_info_path):
        raise FileNotFoundError(f"collect_historical_info.py not found at: {hist_info_path}")

    spec = importlib.util.spec_from_file_location("collect_historical_info", hist_info_path)
    hist_info_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hist_info_module)

    GitRepoManager = hist_info_module.GitRepoManager
    PythonEntityExtractor = hist_info_module.PythonEntityExtractor
    CommitAnalyzer = hist_info_module.CommitAnalyzer
    HistoricalInfoCollector = hist_info_module.HistoricalInfoCollector

    # Similarly handle domain_knowledge_enhancement.py
    domain_enh_path = os.path.join(os.path.dirname(__file__), 'domain_knowledge_enhancement.py')
    if not os.path.exists(domain_enh_path):
        raise FileNotFoundError(f"domain_knowledge_enhancement.py not found at: {domain_enh_path}")

    spec = importlib.util.spec_from_file_location("domain_knowledge_enhancement", domain_enh_path)
    domain_enh_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(domain_enh_module)

    extract_code_entities = domain_enh_module.extract_code_entities
    filter_overlapping_tokens = domain_enh_module.filter_overlapping_tokens
    is_low_quality_commit = domain_enh_module.is_low_quality_commit
    compute_tau = domain_enh_module.compute_tau
    filter_co_modifications = domain_enh_module.filter_co_modifications
    get_co_modification_dict = domain_enh_module.get_co_modification_dict

except Exception as e:
    # Raise directly; do not use placeholder classes
    raise RuntimeError(
        f"Failed to import required modules for online domain knowledge collection.\n"
        f"Error: {e}\n"
        f"Please ensure these files exist in util/ directory:\n"
        f"  - collect_historical_info.py\n"
        f"  - domain_knowledge_enhancement.py"
    ) from e


class OnlineDomainKnowledgeCollector:
    """
    Online Domain Knowledge collector

    For a small set of test functions filtered by BM25, collect in real time:
    1) Historical edit information (reuse HistoricalInfoCollector)
    2) Domain knowledge tokens (reuse logic from domain_knowledge_enhancement)
    """

    # Class-level cache to avoid cloning the repo repeatedly
    _repo_cache = {}  # {(repo, commit): repo_path}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        instance_id: str,
        repo: str,
        base_commit: str,
        logger: logging.Logger,
        temp_dir: Optional[str] = None,
        existing_repo_path: Optional[str] = None,
        dk_log_dir: Optional[str] = None  # New: domain knowledge log directory
    ):
        """
        Initialize the online collector

        Args:
            instance_id: Instance ID
            repo: GitHub repository path (e.g., "scikit-learn/scikit-learn")
            base_commit: Base commit hash
            logger: Logger
            temp_dir: Temporary directory (used for cloning)
            existing_repo_path: Existing repo path (if provided, use it with higher priority)
            dk_log_dir: Domain knowledge log directory
        """
        self.instance_id = instance_id
        self.repo = repo
        self.base_commit = base_commit
        self.logger = logger
        self.existing_repo_path = Path(existing_repo_path) if existing_repo_path else None

        # Initialize temp directory
        if temp_dir is None:
            temp_dir = '/tmp/swe_bench_repos'
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # Repo path (lazy clone)
        self.repo_path = None
        self._collector = None

        # Configure domain knowledge logging (capture module-level logging.xxx() calls in collect_historical_info.py)
        self._dk_log_dir = dk_log_dir
        self._root_handler = None
        self._removed_handlers = []
        self._setup_dk_logging()

    def _setup_dk_logging(self):
        """
        Configure the root logger to write to a dedicated domain knowledge log file.

        This prevents module-level logging.xxx() calls in collect_historical_info.py from polluting the terminal.
        """
        root_logger = logging.getLogger()

        # 1) Remove all StreamHandlers (save them for restoration later)
        for h in root_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root_logger.removeHandler(h)
                self._removed_handlers.append(h)

        # 2) Add a FileHandler to a dedicated log file
        if self._dk_log_dir:
            os.makedirs(self._dk_log_dir, exist_ok=True)
            log_file = os.path.join(self._dk_log_dir, f'{self.instance_id}.log')
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(fh)
            self._root_handler = fh
        else:
            # If no log dir is specified, add a NullHandler to avoid terminal output
            nh = logging.NullHandler()
            root_logger.addHandler(nh)
            self._root_handler = nh

        # 3) Set root logger level
        root_logger.setLevel(logging.DEBUG)

    def _restore_logging(self):
        """Restore the root logger configuration."""
        root_logger = logging.getLogger()

        # Remove the handler we added
        if self._root_handler and self._root_handler in root_logger.handlers:
            root_logger.removeHandler(self._root_handler)
            if isinstance(self._root_handler, logging.FileHandler):
                self._root_handler.close()

        # Restore the original StreamHandlers
        for h in self._removed_handlers:
            if h not in root_logger.handlers:
                root_logger.addHandler(h)

        self._root_handler = None
        self._removed_handlers = []

    def _get_or_clone_repo(self) -> Optional[Path]:
        """
        Get or clone a repo (priority: provided path -> in-memory cache -> disk check -> clone).
        """
        cache_key = (self.repo, self.base_commit)

        with self._cache_lock:
            # ========== Priority 1: Use the provided existing_repo_path ==========
            if self.existing_repo_path and self.existing_repo_path.exists():
                self.logger.info(f"Using pre-cloned repo from caller: {self.existing_repo_path}")
                self._repo_cache[cache_key] = self.existing_repo_path
                return self.existing_repo_path

            # ========== Priority 2: Check in-memory cache ==========
            if cache_key in self._repo_cache:
                cached_path = self._repo_cache[cache_key]
                if cached_path and cached_path.exists():
                    self.logger.info(f"Using cached repo from memory: {cached_path}")
                    return cached_path

            # ========== Priority 3: Check if it already exists on disk ==========
            # Use instance_id directly as the directory name
            potential_path = Path(self.temp_dir) / self.instance_id

            if potential_path.exists():
                try:
                    import subprocess
                    result = subprocess.run(
                        ['git', 'rev-parse', 'HEAD'],
                        cwd=potential_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    current_commit = result.stdout.strip()

                    if current_commit == self.base_commit:
                        self.logger.info(f"Found existing repo on disk: {potential_path}")
                        self._repo_cache[cache_key] = potential_path
                        return potential_path
                    else:
                        self.logger.debug(f"Repo exists but wrong commit: {potential_path}")
                except Exception as e:
                    self.logger.debug(f"Failed to verify repo at {potential_path}: {e}")

            # ========== Priority 4: Clone ==========
            self.logger.info(f"No reusable repo found, cloning: {self.repo} @ {self.base_commit}")

            git_manager = GitRepoManager(self.temp_dir, self.instance_id)
            repo_path = git_manager.clone_repo(self.repo, self.base_commit)

            if repo_path:
                self._repo_cache[cache_key] = repo_path
                self.logger.info(f"Repository cloned to: {repo_path}")
            else:
                self.logger.error("Failed to clone repository")

            return repo_path

    def _collect_historical_info_for_test(
        self,
        test_function: str,
        coverage_graph: Dict
    ) -> Dict:
        """
        Collect historical info for a single test function (reuse existing logic).

        Args:
            test_function: Test function name (format: file::function)
            coverage_graph: Coverage graph data

        Returns:
            A dict of historical info
        """
        if self.repo_path is None:
            self.repo_path = self._get_or_clone_repo()
            if self.repo_path is None:
                return {}

        # Initialize the collector (reuse existing class)
        if self._collector is None:
            # Create temporary output directories
            temp_output_dir = os.path.join(self.temp_dir, f'temp_hist_info_{self.instance_id}')
            temp_log_dir = os.path.join(self.temp_dir, f'temp_logs_{self.instance_id}')

            self._collector = HistoricalInfoCollector(
                temp_output_dir,
                temp_log_dir,
                self.instance_id
            )

        # Get covered entities
        covered_entities = coverage_graph.get(test_function, {}).get('nodes', [])

        if not covered_entities:
            self.logger.warning(f"No coverage data for test: {test_function}")
            return {}

        # Call the existing collection method
        try:
            result = self._collector.collect_for_test(
                self.logger,
                self.repo_path,
                test_function,
                covered_entities,
                self.base_commit
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to collect history for {test_function}: {e}")
            return {}

    def _extract_domain_knowledge(
        self,
        historical_info: Dict,
        test_function: str,
        min_tau: int = 3,
        auto_threshold: bool = False
    ) -> List[str]:
        """
        Extract domain knowledge tokens from historical info (reuse existing logic).

        Args:
            historical_info: Historical info dict
            test_function: Test function name
            min_tau: Minimum frequency threshold
            auto_threshold: Whether to automatically compute the threshold

        Returns:
            A list of domain knowledge tokens
        """
        if not historical_info:
            return []

        all_candidate_tokens = []

        # ========== Handle init_commit ==========
        init_commit = historical_info.get("init_commit", None)
        if init_commit:
            commit_msg = init_commit.get("commit_message", "")
            if not is_low_quality_commit(commit_msg, self.logger):
                tokens = extract_code_entities(commit_msg, "init_commit", self.logger)
                all_candidate_tokens.extend(tokens)
            else:
                self.logger.debug("  init_commit filtered (low quality)")

        # ========== Handle co_modifications ==========
        co_mod_dict = get_co_modification_dict(test_function, historical_info)
        self.logger.debug(f"  co_modify entries: {len(co_mod_dict)}")

        if co_mod_dict:
            # Compute threshold
            tau = compute_tau(co_mod_dict, min_tau, auto_threshold, self.logger)

            # Frequency filter
            filtered_co_mod = filter_co_modifications(co_mod_dict, tau, self.logger)

            # Extract tokens from entity names
            for entity in filtered_co_mod.keys():
                tokens = extract_code_entities(entity, "co_modify", self.logger)
                all_candidate_tokens.extend(tokens)

        # ========== Deduplicate (token-level) ==========
        seen = set()
        unique_tokens = []
        for t in all_candidate_tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        self.logger.debug(f"  Candidate tokens deduplicated: {len(all_candidate_tokens)} -> {len(unique_tokens)}")

        # ========== Deduplicate against test_func ==========
        final_tokens = filter_overlapping_tokens(unique_tokens, test_function, self.logger)

        self.logger.info(f"  [{test_function.split('::')[-1]}] Final tokens: {len(final_tokens)}")
        if final_tokens:
            self.logger.debug(f"    Tokens: {final_tokens}")

        return final_tokens

    def collect_for_tests(
        self,
        test_functions: List[str],
        coverage_graph: Dict,
        min_tau: int = 3,
        auto_threshold: bool = False
    ) -> Dict[str, List[str]]:
        """
        Collect domain knowledge for a list of tests.

        Args:
            test_functions: List of test function names
            coverage_graph: Coverage graph data (format: {test_func: {'nodes': [...]}})
            min_tau: Minimum frequency threshold
            auto_threshold: Whether to automatically compute the threshold

        Returns:
            {test_func: [domain_knowledge_tokens], ...}
        """
        self.logger.info(f"Starting online domain knowledge collection for {len(test_functions)} tests")

        result = {}

        for idx, test_func in enumerate(test_functions, 1):
            self.logger.info(f"Processing [{idx}/{len(test_functions)}]: {test_func}")

            try:
                # Step 1: Collect historical info
                hist_info = self._collect_historical_info_for_test(test_func, coverage_graph)

                # Step 2: Extract domain knowledge
                tokens = self._extract_domain_knowledge(
                    hist_info,
                    test_func,
                    min_tau,
                    auto_threshold
                )

                result[test_func] = tokens

            except Exception as e:
                self.logger.error(f"Failed to process {test_func}: {e}", exc_info=True)
                result[test_func] = []

        # Summary stats
        total_tokens = sum(len(tokens) for tokens in result.values())
        non_empty = sum(1 for tokens in result.values() if tokens)
        self.logger.info(
            f"Online collection complete: {non_empty}/{len(result)} tests have domain knowledge, {total_tokens} total tokens"
        )

        return result

    def cleanup(self):
        """
        Cleanup resources.

        Note: Do not delete cached repositories so they can be reused by later instances.
        """
        # Cleanup temporary files
        temp_output_dir = os.path.join(self.temp_dir, f'temp_hist_info_{self.instance_id}')
        temp_log_dir = os.path.join(self.temp_dir, f'temp_logs_{self.instance_id}')

        for dir_path in [temp_output_dir, temp_log_dir]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    self.logger.debug(f"Cleaned up: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {dir_path}: {e}")

        # Restore root logger configuration
        self._restore_logging()

        self.logger.info("Online collector cleanup complete")

    @classmethod
    def clear_repo_cache(cls):
        """Clear the repo cache (optional, used to free disk space)."""
        with cls._cache_lock:
            for repo_path in cls._repo_cache.values():
                if repo_path and repo_path.exists():
                    try:
                        shutil.rmtree(repo_path)
                    except Exception:
                        pass
            cls._repo_cache.clear()


# ==================== Convenience function ====================

def collect_online_domain_knowledge(
    instance_id: str,
    repo: str,
    base_commit: str,
    test_functions: List[str],
    coverage_graph: Dict,
    logger: logging.Logger,
    min_tau: int = 3,
    auto_threshold: bool = False
) -> Dict[str, List[str]]:
    """
    Convenience function: collect domain knowledge for a list of tests.

    Args:
        instance_id: Instance ID
        repo: GitHub repository path
        base_commit: Base commit
        test_functions: List of test function names
        coverage_graph: Coverage graph data
        logger: Logger
        min_tau: Minimum frequency threshold
        auto_threshold: Whether to automatically compute the threshold

    Returns:
        {test_func: [tokens], ...}
    """
    collector = OnlineDomainKnowledgeCollector(
        instance_id,
        repo,
        base_commit,
        logger
    )

    try:
        return collector.collect_for_tests(
            test_functions,
            coverage_graph,
            min_tau,
            auto_threshold
        )
    finally:
        collector.cleanup()

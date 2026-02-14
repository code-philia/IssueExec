# Localizer.py
import os
import json
import ijson
import logging
from pathlib import Path

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from util.compress_file import get_skeleton
from util.postprocess_data import extract_code_blocks, extract_locs_for_files
from util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

import ast
import re
from rank_bm25 import BM25Okapi
from prompt import *


MAX_CONTEXT_LENGTH = 60000 # < 65536


SUSPICIOUS_LOCATION_RETRY_PREFIXES = [
    # Retry 1: light constraints
    """[RETRY NOTICE] Your previous response contained locations not found in the codebase. 
Please select ONLY from:
1. Locations in "Code Locations Covered by Related Tests"
2. Locations visible in the traceback that exist in the project

""",

    # Retry 2: stronger constraints with explicit candidate list
    """[STRICT MODE] WARNING: You previously output invalid code locations.

Here are ALL valid locations you can choose from:
{valid_candidate_list}

You MUST select ONLY from the above list. Do NOT invent location names.

""",

    # Retry 3: strict copy-only mode
    """[FINAL ATTEMPT] CRITICAL: Your output contained invalid locations.

MANDATORY: You can ONLY output locations from this EXACT list:
{valid_candidate_list}

Copy location names CHARACTER-FOR-CHARACTER from the list above.

"""
]


def normalize_traceback_path(abs_path: str, project_file_paths: List[str]) -> Optional[str]:
    """
    Map an absolute traceback path to a repository-relative path.
    """
    # Normalize separators for cross-platform traceback strings.
    abs_path = abs_path.replace('\\', '/')
    
    for proj_path in project_file_paths:
        # Suffix match: traceback path usually ends with project-relative path.
        if abs_path.endswith('/' + proj_path) or abs_path == proj_path:
            return proj_path
    
    return None


def validate_location_in_project(
    loc: str,
    files: List[Tuple[str, Any]],
    classes: List[Dict],
    functions: List[Dict]
) -> Tuple[bool, Optional[str]]:
    """
    Validate whether a location string points to a real entity in the project.
    
    Args:
        loc: location string, for example:
             - file_path::function_name
             - file_path::ClassName
             - file_path::ClassName.method_name
             - file_path::ClassName::method_name
        files: project files
        classes: parsed classes
        functions: parsed top-level functions
        
    Returns:
        (is_valid, normalized_loc): validity flag and normalized location
    """
    if '::' not in loc:
        return False, None
    
    # Parse location.
    parts = loc.split('::', 1)
    file_path = parts[0]
    identifier = parts[1] if len(parts) > 1 else ''
    
    # Check that file exists.
    file_paths = [f[0] for f in files]
    if file_path not in file_paths:
        return False, None
    
    # Parse identifier part (supports both `::` and `.` method separators).
    class_name = None
    method_name = None
    
    if '::' in identifier:
        # Format: file::ClassName::method_name
        sub_parts = identifier.split('::', 1)
        class_name = sub_parts[0]
        method_name = sub_parts[1] if len(sub_parts) > 1 else None
    elif '.' in identifier:
        # Format: file::ClassName.method_name
        sub_parts = identifier.split('.', 1)
        class_name = sub_parts[0]
        method_name = sub_parts[1] if len(sub_parts) > 1 else None
    else:
        # Format: file::identifier (function/class/method name)
        identifier_name = identifier
        
        # Try top-level function first.
        for func in functions:
            if func.get('file') == file_path and func.get('name') == identifier_name:
                return True, loc
        
        # Then try class name.
        for cls in classes:
            if cls.get('file') == file_path and cls.get('name') == identifier_name:
                return True, loc
        
        # Finally try method name only (common in traceback frames).
        for cls in classes:
            if cls.get('file') == file_path:
                for method in cls.get('methods', []):
                    if method.get('name') == identifier_name:
                        # Return normalized full location.
                        normalized = f"{file_path}::{cls['name']}.{identifier_name}"
                        return True, normalized
        
        return False, None
    
    # Handle explicit class-method form.
    if class_name and method_name:
        for cls in classes:
            if cls.get('file') == file_path and cls.get('name') == class_name:
                for method in cls.get('methods', []):
                    if method.get('name') == method_name:
                        # Normalize to dot style.
                        normalized = f"{file_path}::{class_name}.{method_name}"
                        return True, normalized
        return False, None
    
    # Class-only case.
    if class_name and not method_name:
        for cls in classes:
            if cls.get('file') == file_path and cls.get('name') == class_name:
                return True, f"{file_path}::{class_name}"
        return False, None
    
    return False, None


def extract_and_validate_locations_from_traceback(
    issue_desc: str,
    files: List[Tuple[str, Any]],
    classes: List[Dict],
    functions: List[Dict],
    logger=None
) -> List[str]:
    """
    Extract traceback frames from issue text, map them to repository paths,
    and return only validated locations.

    Returns locations in format: ["file_path::entity_name", ...]
    """
    import re
    
    # Standard traceback frame: File "xxx.py", line N, in func_name
    pattern = r'File\s+"([^"]+)",\s+line\s+\d+,\s+in\s+(\S+)'
    matches = re.findall(pattern, issue_desc)
    
    if not matches:
        if logger:
            logger.info("No traceback entries found in issue description")
        return []
    
    if logger:
        logger.info(f"Found {len(matches)} traceback entries")
    
    project_file_paths = [f[0] for f in files]
    validated_locations = []
    seen = set()  # Deduplicate candidate locations.
    
    for abs_path, func_name in matches:
        # Skip stdlib frames.
        if 'python' in abs_path.lower() and 'site-packages' not in abs_path.lower():
            continue
        
        # Map absolute frame path to project path.
        proj_path = normalize_traceback_path(abs_path, project_file_paths)
        if not proj_path:
            if logger:
                logger.debug(f"Could not map traceback path to project: {abs_path}")
            continue
        
        # Build candidate location.
        candidate_loc = f"{proj_path}::{func_name}"
        
        if candidate_loc in seen:
            continue
        seen.add(candidate_loc)
        
        # Validate against parsed repository structure.
        is_valid, normalized_loc = validate_location_in_project(
            candidate_loc, files, classes, functions
        )
        
        if is_valid and normalized_loc:
            validated_locations.append(normalized_loc)
            if logger:
                logger.debug(f"Validated traceback location: {normalized_loc}")
        else:
            if logger:
                logger.debug(f"Could not validate traceback location: {candidate_loc}")
    
    # Keep order while deduplicating.
    unique_locations = list(dict.fromkeys(validated_locations))
    
    if logger:
        logger.info(f"Validated {len(unique_locations)} locations from traceback")
    
    return unique_locations


def should_retry_suspicious_locations(
    all_locations: List[str],
    valid_locations: List[str]
) -> bool:
    """
    Decide whether suspicious-location localization should retry.
    Trigger when no valid location is found, or invalid ratio >= 50%.
    """
    if len(all_locations) == 0:
        return False
    
    invalid_count = len(all_locations) - len(valid_locations)
    if invalid_count == 0:
        return False
    
    # Retry when invalid count is at least half.
    threshold = max(1, len(all_locations) // 2)
    return invalid_count >= threshold



def extract_code_references_from_issue_description(
    issue_desc: str,
    files: List[Tuple[str, Any]],
    classes: List[Dict],
    functions: List[Dict],
    logger=None
) -> List[str]:
    """
    Extract direct code references from issue text (files/lines/functions).
    Used as fallback when coverage-based localization fails.

    Strategy:
    1. Regex match for Python file paths.
    2. Resolve line numbers to entities via AST.
    3. Validate against repository structure.
    
    Args:
        issue_desc: issue description text
        files: project files [(file_path, content), ...]
        classes: parsed classes
        functions: parsed functions
        logger: logger instance
        
    Returns:
        Validated locations: ["file_path::entity_name", ...]
    """
    import re
    import ast
    
    if logger:
        logger.info("Starting Issue-Direct extraction fallback")
    
    # Build project-path and content indexes.
    project_file_paths = set(f[0] for f in files)
    file_content_map = {f[0]: f[1] for f in files}
    
    extracted_locations = []
    seen = set()
    
    # Pattern 1: file path + line number.
    # Example: "...admin_modify.py ... line 102"
    file_line_pattern = r'["\']?([\w/]+\.py)["\']?[^0-9]*(?:line\s*|:)(\d+)'
    
    # Pattern 2: file path only.
    file_only_pattern = r'["\']?([\w/]+\.py)["\']?'
    
    # Pattern 3: quoted function names (weak signal, reserved for future use).
    func_pattern = r'["\'](\w+)["\']'
    
    # First try file+line matches.
    file_line_matches = re.finditer(file_line_pattern, issue_desc, re.IGNORECASE)
    
    for match in file_line_matches:
        file_path_candidate = match.group(1)
        line_number = int(match.group(2))
        
        # Validate file path against project files.
        matched_file_path = None
        for proj_path in project_file_paths:
            # Allow partial-path match from issue text.
            if proj_path.endswith(file_path_candidate) or file_path_candidate.endswith(proj_path) or proj_path == file_path_candidate:
                matched_file_path = proj_path
                break
        
        if not matched_file_path:
            if logger:
                logger.debug(f"File path not found in project: {file_path_candidate}")
            continue
        
        # Try to resolve entity by line number.
        entity_name = _get_entity_at_line(
            file_content_map.get(matched_file_path),
            line_number,
            logger
        )
        
        if entity_name:
            location = f"{matched_file_path}::{entity_name}"
            if location not in seen:
                seen.add(location)
                extracted_locations.append(location)
                if logger:
                    logger.info(f"Issue-Direct extracted (file+line): {location}")
        else:
            # Fallback to file-level location.
            location = matched_file_path
            if location not in seen:
                seen.add(location)
                extracted_locations.append(location)
                if logger:
                    logger.info(f"Issue-Direct extracted (file-level fallback): {location}")
    
    # If file+line failed, try file-only matches.
    if not extracted_locations:
        file_only_matches = re.finditer(file_only_pattern, issue_desc)
        
        for match in file_only_matches:
            file_path_candidate = match.group(1)
            
            # Validate file path.
            matched_file_path = None
            for proj_path in project_file_paths:
                if proj_path.endswith(file_path_candidate) or file_path_candidate.endswith(proj_path) or proj_path == file_path_candidate:
                    matched_file_path = proj_path
                    break
            
            if matched_file_path and matched_file_path not in seen:
                seen.add(matched_file_path)
                extracted_locations.append(matched_file_path)
                if logger:
                    logger.info(f"Issue-Direct extracted (file-only): {matched_file_path}")
    
    if logger:
        logger.info(f"Issue-Direct extraction complete: {len(extracted_locations)} locations found")
    
    return extracted_locations


def _get_entity_at_line(
    file_content: Union[str, List[str], None],
    line_number: int,
    logger=None
) -> Optional[str]:
    """
    Resolve a line number to the innermost containing class/function entity.
    
    Args:
        file_content: file content (string or line list)
        line_number: target line number (1-based)
        logger: logger instance
        
    Returns:
        Entity name, or None when not found
    """
    import ast
    
    if not file_content:
        return None
    
    try:
        # Ensure content is string.
        if isinstance(file_content, list):
            content_str = '\n'.join(file_content)
        else:
            content_str = file_content
        
        tree = ast.parse(content_str)
        
        # Collect entities with line ranges.
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                end_line = getattr(node, 'end_lineno', node.lineno + 1)
                # Detect whether function is a class method.
                parent_class = _find_parent_class(tree, node)
                if parent_class:
                    entity_name = f"{parent_class}.{node.name}"
                else:
                    entity_name = node.name
                entities.append((node.lineno, end_line, entity_name))
                
            elif isinstance(node, ast.ClassDef):
                end_line = getattr(node, 'end_lineno', node.lineno + 1)
                entities.append((node.lineno, end_line, node.name))
        
        entities.sort(key=lambda x: x[0])
        
        best_match = None
        best_range = float('inf')
        
        for start_line, end_line, name in entities:
            if start_line <= line_number <= end_line:
                range_size = end_line - start_line
                if range_size < best_range:
                    best_range = range_size
                    best_match = name
        
        if best_match and logger:
            logger.debug(f"Line {line_number} maps to entity: {best_match}")
        
        return best_match
        
    except SyntaxError as e:
        if logger:
            logger.warning(f"Syntax error parsing file for line mapping: {e}")
        return None
    except Exception as e:
        if logger:
            logger.warning(f"Error mapping line to entity: {e}")
        return None


def _find_parent_class(tree: ast.AST, target_node: ast.FunctionDef) -> Optional[str]:
    import ast
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if child is target_node:
                    return node.name

            for child in ast.walk(node):
                if child is target_node and child is not node:
                    return node.name
    return None



# testEN（extract_key_words + filter_test_cases_by_token_match）
def extract_keywords(issue_desc):
    import re
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 
                  'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                  'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 
                  'these', 'those', 'when', 'where', 'why', 'how', 'if', 'then', 'get', 
                  'got', 'some', 'each', 'while', 'even', 'though', 'def'

                  'import', 'from', 'return', 'class', 'elif', 'else', 'pass',
                  'break', 'continue', 'description', 'issue', 'example',
                  
                  'test', 'tests', 'testing', 'pytest', 'unittest',
                  'py', 'pyc', 'pyx', 'pyd', 
                  'init', 'main', 'setup', 
                  'util', 'utils', 
                  'common', 'base', 'core', 
                  'lib', 'libs', 'src', 


                  'dot', 'dotprint', 'dotnode',
                  }
    
                  
    def remove_common_suffixes(word):
        """
        Remove common English suffixes and return a simple stem.
        """

        suffixes = ['ing', 'ed', 'es', 's', 'er', 'ly', 'tion', 'ment']
        
        original = word
        for suffix in suffixes:
            if len(word) > len(suffix) + 2: 
                if word.endswith(suffix):
                    stem = word[:-len(suffix)]
                    if len(stem) >= 3:
                        return stem, original
        return original, None

    def split_compound_word(word):
        parts = word.split('_')
        result = []
        
        for part in parts:
            camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            result.extend(camel_split.lower().split())
        
        return result
    
    clean_text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', issue_desc)
    words = clean_text.split()
    

    all_keywords = []
    for word in words:
        if len(word) > 2:

            sub_words = split_compound_word(word)
            for sub_word in sub_words:

                stem, original = remove_common_suffixes(sub_word)
                all_keywords.append(stem)
                if original and original != stem:
                    all_keywords.append(original)
    

    keywords = [word for word in all_keywords 
                if len(word) > 2 and word.lower() not in stop_words]
    

    result = []
    for word in keywords:
        result.append(word.lower())
    
    return result



def filter_test_cases_by_token_match(
    issue_desc: str,
    test_cases: List,
    min_tests: int = 10,
    max_tests: int = 200,
    random_backfill: bool = False,
    strict_mode: bool = False
) -> List:
    from rank_bm25 import BM25Okapi
    import re
    import random
    
    if not test_cases:
        return []
    

    def tokenize(text):
        return extract_keywords(text)
    

    corpus = []
    for test in test_cases:

        test_text = test['name'] + " " + test['name'] + " " + (test['docstring'] or "")
        corpus.append(tokenize(test_text))
    
    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(issue_desc)
    
    scores = bm25.get_scores(query_tokens)
    
    test_score_pairs = sorted(zip(test_cases, scores), key=lambda x: x[1], reverse=True)
    
    threshold = 0.5 if strict_mode else 0.01
    
    matched_tests = [test for test, score in test_score_pairs if score > threshold]
    unmatched_tests = [test for test, score in test_score_pairs if score <= threshold]
    
    if len(matched_tests) < min_tests and random_backfill and unmatched_tests:

        needed = min(min_tests - len(matched_tests), len(unmatched_tests))
        matched_tests.extend(random.sample(unmatched_tests, needed))
    
    return matched_tests[:max_tests]



class BaseLocalizer(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

        # Fallback: discover from repository structure
        self.files, self.classes, self.functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass



class RelatedTestRetriever(BaseLocalizer):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        coverage_graph_path: Optional[str] = None,
        test_functions_path: Optional[str] = None,
        expand_query: bool = True,
        domain_knowledge_path: Optional[str] = None,
        use_online_domain_knowledge: bool = False,
        repo: Optional[str] = None,
        base_commit: Optional[str] = None,
        repo_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 3000
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.coverage_graph_path = os.path.join(coverage_graph_path, instance_id + ".json") if coverage_graph_path else None
        self.test_functions_path = test_functions_path
        

        self.use_online_domain_knowledge = use_online_domain_knowledge
        self.repo = repo
        self.base_commit = base_commit
        self.repo_path = repo_path  
        
        if use_online_domain_knowledge:

            self.domain_knowledge_path = None
            self.logger.info("Online domain knowledge mode enabled - will collect after BM25 filtering")
        else:

            self.domain_knowledge_path = os.path.join(domain_knowledge_path, instance_id + ".json") if domain_knowledge_path else None
        
        self.obtain_relevant_tests_prompt = obtain_relevant_tests_prompt

        self.logger.info(f"=== Initializing RelatedTestRetriever for {instance_id} ===")
        self.logger.info(f"Coverage graph path: {self.coverage_graph_path}")
        self.logger.info(f"Test functions path: {self.test_functions_path}")
        self.logger.info(f"Total files in structure: {len(self.files)}")

        # Load coverage data and test functions
        self.coverage_data = self._load_coverage_data()
        self.logger.info(f"Coverage data loaded: {len(self.coverage_data)} entries")

        self.test_functions = self._load_test_functions()
        self.logger.info(f"Test functions from file: {len(self.test_functions)} functions")
        
        self.test_functions = self._discover_test_functions()
        self.logger.info(f"Test functions after discovery: {len(self.test_functions)} functions")

        # query expansion
        if expand_query:
            self.problem_statement = self.query_expansion(problem_statement)

        # ENtoken_matchENtest_functions
        before_filter = len(self.test_functions)
        self.test_functions = filter_test_cases_by_token_match(issue_desc=self.problem_statement, test_cases=self.test_functions, min_tests=10, max_tests=200, random_backfill=True, strict_mode=False)
        self.logger.info(f"Test functions after BM25 filtering: {len(self.test_functions)} (from {before_filter})")

        # Load domain knowledge
        if not self.use_online_domain_knowledge:
            self.domain_knowledge = self._load_domain_knowledge()
            self._merge_domain_knowledge()
            if self.domain_knowledge:
                self.logger.info(f"Domain knowledge loaded from file: {len(self.domain_knowledge)} entries")
        else:
            self.domain_knowledge = {}
            self.logger.info("Starting online domain knowledge collection for BM25-filtered tests...")
            
            if len(self.test_functions) > 0:
                try:
                    from util.domain_knowledge_utils import OnlineDomainKnowledgeCollector
                    
                    dk_log_dir = None
                    for handler in self.logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            log_path = Path(handler.baseFilename)
                            dk_log_dir = str(log_path.parent.parent / 'domain_knowledge_logs')
                            break
                    
                    collector = OnlineDomainKnowledgeCollector(
                        instance_id=self.instance_id,
                        repo=self.repo,
                        base_commit=self.base_commit,
                        logger=self.logger,
                        existing_repo_path=self.repo_path,
                        dk_log_dir=dk_log_dir
                    )
                    
                    test_names = [tf['name'] for tf in self.test_functions]
                    
                    if getattr(self, '_use_lazy_loading', False):
                        coverage_for_dk = self._load_coverage_for_tests(test_names)
                        self.coverage_data.update(coverage_for_dk)
                    else:
                        coverage_for_dk = self.coverage_data
                    
                    domain_knowledge_map = collector.collect_for_tests(
                        test_names,
                        coverage_for_dk
                    )
                    
                    for test in self.test_functions:
                        test_name = test['name']
                        test['domain_tokens'] = domain_knowledge_map.get(test_name, [])
                    
                    with_tokens = sum(1 for tf in self.test_functions if tf.get('domain_tokens'))
                    total_tokens = sum(len(tf.get('domain_tokens', [])) for tf in self.test_functions)
                    
                    self.logger.info(f"Online domain knowledge collection complete:")
                    self.logger.info(f"  - Tests with tokens: {with_tokens}/{len(self.test_functions)}")
                    self.logger.info(f"  - Total tokens: {total_tokens}")
                    
                    collector.cleanup()
                    
                except ImportError as e:
                    self.logger.error(f"Failed to import online collector: {e}")
                    for test in self.test_functions:
                        test['domain_tokens'] = []
                        
                except Exception as e:
                    self.logger.error(f"Online domain knowledge collection failed: {e}", exc_info=True)
                    for test in self.test_functions:
                        test['domain_tokens'] = []
            else:
                self.logger.warning("No test functions after BM25 filtering, skipping domain knowledge collection")

        self.logger.info(f"=== RelatedTestRetriever initialization complete ===")

        if len(self.test_functions) == 0 and len(self.coverage_data) > 0:
            self.logger.warning("No test functions found but coverage data exists, running diagnosis...")
            self._diagnose_coverage_mismatch()



    def query_expansion(self, problem_statement: str) -> str:
        """
        Expand the problem statement by extracting entities and their variations
        to improve test retrieval accuracy.
        
        Args:
            problem_statement: Original issue description
            
        Returns:
            expanded_problem_statement: Original statement + expanded terms
        """
        from util.api_requests import num_tokens_from_messages
        from util.model import make_model
        
        # Construct the prompt message
        message = query_expansion_prompt.format(
            problem_statement=problem_statement
        ).strip()
        
        self.logger.info("Performing query expansion on problem statement")
        
        # Check if message is too long
        if num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH:
            self.logger.warning("Query expansion prompt too long, using original problem statement")
            return problem_statement
        
        try:
            # Create model and generate expansion
            model = make_model(
                model=self.model_name,
                backend=self.backend,
                logger=self.logger,
                max_tokens=1000,  # Limit expansion output
                temperature=0,
                batch_size=1,
            )
            
            traj = model.codegen(message, num_samples=1)[0]
            raw_output = traj["response"]
            
            self.logger.info(f"Query expansion raw output:\n{raw_output}")
            
            # Extract the structured expansion content
            expanded_content = self._parse_expansion_output(raw_output)
            
            if not expanded_content:
                self.logger.warning("No expanded content extracted, using original problem statement")
                return problem_statement
            
            # Simple concatenation
            expanded_statement = f"{problem_statement}\n\n## Expanded Query Terms:\n{expanded_content}"
            
            self.logger.info(f"Query expansion completed. Original length: {len(problem_statement)}, "
                            f"Expanded length: {len(expanded_statement)}")
            
            return expanded_statement
            
        except Exception as e:
            self.logger.error(f"Error during query expansion: {e}")
            self.logger.warning("Falling back to original problem statement")
            return problem_statement


    def _parse_expansion_output(self, raw_output: str) -> str:
        """
        Extract the structured expansion content from code blocks.
        
        Args:
            raw_output: Raw model output
            
        Returns:
            Extracted expansion content (empty string if extraction fails)
        """
        import re
        
        # Extract content from code blocks
        match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            self.logger.debug(f"Extracted expansion content:\n{content}")
            return content
        else:
            self.logger.warning("No code block found in expansion output")
            return ""


    def localize(self, top_n=5, mock=False) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Localize relevant test functions based on the issue description.
        
        Returns:
            found_tests: List of relevant test function names
            metadata: Dictionary containing intermediate results
            traj: Trajectory information for the LLM call
        """

        from util.api_requests import num_tokens_from_messages
        from util.model import make_model

        # Get available test functions
        if not self.test_functions:
            self.logger.warning("No test functions found")
            return [], {"raw_output_tests": "", "found_tests": []}, {}
        
        # Format test functions for prompt
        test_functions_text = self._format_test_functions_for_prompt(self.test_functions)

        # Conditionally prepend Related Concepts note if any test has domain_tokens
        has_domain_tokens = any(
            test.get('domain_tokens') 
            for test in self.test_functions
        )
        
        if has_domain_tokens:
            related_concepts_note = """Note: Some tests include "Related Concepts" derived from historical code changes and commit analysis. These concepts may indicate:
- Terminology expansions (e.g., abbreviations to full names)
- Frequently co-modified modules
- Hidden dependencies between components
Treat these as supplementary hints rather than definitive evidence.

"""
            test_functions_text = related_concepts_note + test_functions_text

        # Create prompt
        message = self.obtain_relevant_tests_prompt.format(
            problem_statement=self.problem_statement,
            test_functions=test_functions_text,
            max_tests=top_n
        ).strip()

        # Create prompt
        message = self.obtain_relevant_tests_prompt.format(
            problem_statement=self.problem_statement,
            test_functions=test_functions_text,
            max_tests=top_n
        ).strip()

        self.logger.info(f"Prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_tests": "", "found_tests": []}, traj

        # Handle context length by batching if necessary
        def message_too_long(message):
            return num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH

        # If message is too long, process in batches
        if message_too_long(message):
            self.logger.info("Message too long, processing in batches")
            
            # ENtokenEN
            def safe_token_count(msg):
                count = num_tokens_from_messages(msg, self.model_name)
                if "deepseek" in self.model_name.lower():
                    count = int(count * 1.3)
                return count
            
            SAFE_MAX_LENGTH = int(MAX_CONTEXT_LENGTH * 0.7) 
            COMPLETION_BUFFER = 1000
            
            # ENbatchEN
            estimated_batches = 8
            batch_size = max(1, len(self.test_functions) // estimated_batches)
            
            self.logger.info(f"Starting with batch size: {batch_size}, safe max length: {SAFE_MAX_LENGTH}")
            
            found_tests = []
            all_trajs = []
            
            i = 0
            while i < len(self.test_functions):
                max_attempts = 5
                current_batch_size = min(batch_size, len(self.test_functions) - i)
                
                for attempt in range(max_attempts):
                    if current_batch_size <= 0:
                        break
                        
                    batch = self.test_functions[i:i + current_batch_size]
                    batch_text = self._format_test_functions_for_prompt(batch)
                    batch_message = self.obtain_relevant_tests_prompt.format(
                        problem_statement=self.problem_statement,
                        test_functions=batch_text,
                        max_tests=top_n
                    ).strip()
                    
                    batch_tokens = safe_token_count(batch_message)
                    total_needed = batch_tokens + COMPLETION_BUFFER
                    
                    self.logger.info(f"Batch attempt {attempt+1} (size={current_batch_size}): {batch_tokens} tokens, total needed: {total_needed}")
                    
                    if total_needed <= SAFE_MAX_LENGTH:
                        break
                    else:
                        self.logger.warning(f"Batch too large ({total_needed} > {SAFE_MAX_LENGTH}), reducing size...")
                        current_batch_size = max(1, current_batch_size // 2)
                        if current_batch_size == 1 and total_needed > SAFE_MAX_LENGTH:
                            self.logger.error(f"Skipping oversized function: {batch[0].get('name', 'unknown')}")
                            i += 1
                            current_batch_size = 0
                            break
                
                if current_batch_size > 0:
                    try:
                        model = make_model(
                            model=self.model_name,
                            backend=self.backend,
                            logger=self.logger,
                            max_tokens=self.max_tokens,
                            temperature=0,
                            batch_size=1,
                        )
                        
                        batch_traj = model.codegen(batch_message, num_samples=1)[0]
                        batch_traj["prompt"] = batch_message
                        all_trajs.append(batch_traj)
                        
                        batch_found_tests = self._parse_model_return_lines(batch_traj["response"])
                        found_tests.extend(batch_found_tests)
                        
                        self.logger.info(f"Batch completed successfully, found {len(batch_found_tests)} tests")
                        i += current_batch_size
                        
                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e}")
                        i += current_batch_size
            
            # Merge trajectories
            traj = {
                "prompt": message,
                "response": [t["response"] for t in all_trajs],
                "usage": {
                    "completion_tokens": sum(t["usage"]["completion_tokens"] for t in all_trajs),
                    "prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in all_trajs),
                },
            }
            raw_output = "\n".join([t["response"] for t in all_trajs])
        else:
            # Single batch processing
            model = make_model(
                model=self.model_name,
                backend=self.backend,
                logger=self.logger,
                max_tokens=self.max_tokens,
                temperature=0,
                batch_size=1,
            )
            
            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]

            found_tests = self._parse_model_return_lines(raw_output)

        # Filter valid test functions
        valid_test_names = {tf['name'] for tf in self.test_functions}
        test_information_dict = dict()
        for test in self.test_functions:
            test_information_dict[test['name']] = [test['docstring'], test['line_number']]

        # === Validation + Retry Policy ===
        def validate_found_tests(found_tests_list):
            """Validate model-returned tests against known candidate tests."""
            valid = []
            invalid = []
            for test in found_tests_list:
                if test in valid_test_names:
                    valid.append(test)
                else:
                    invalid.append(test)
            return valid, invalid

        def log_validation_details(found_tests_list, valid_tests, invalid_tests, attempt_num):
            """Log validation details for each attempt."""
            self.logger.info(f"=== Validation Details (Attempt {attempt_num}) ===")
            self.logger.info(f"LLM returned {len(found_tests_list)} test(s): {found_tests_list}")
            self.logger.info(f"Valid tests ({len(valid_tests)}): {valid_tests}")
            if invalid_tests:
                self.logger.warning(f"Invalid tests ({len(invalid_tests)}): {invalid_tests}")
                for inv_test in invalid_tests:
                    # Explain likely source of invalid names.
                    if inv_test in self.problem_statement:
                        reason = "Likely from Issue Description (proposed new test)"
                    else:
                        reason = "Not in candidate test list"
                    self.logger.warning(f"  - INVALID: {inv_test} | Reason: {reason}")

        def should_retry(found_tests_list, invalid_tests):
            """Retry when invalid outputs indicate likely hallucinations."""
            # No invalid tests -> no retry.
            if not invalid_tests:
                return False
            # Retry if invalid count >= max(1, 50% of all returned tests).
            threshold = max(1, len(found_tests_list) // 2)
            return len(invalid_tests) >= threshold

        # Progressive retry prefixes.
        RETRY_PREFIXES = [
            # Retry 1: light constraints
            """[RETRY NOTICE] Your previous response contained tests not in the provided list. Please select ONLY from the "Test Functions" section below. Do not reference any test names mentioned in the Issue Description.

""",
            # Retry 2: strict constraints
            """[STRICT MODE] WARNING: You previously output invalid test names. You MUST:
1. Select ONLY from tests listed in "Test Functions" section below
2. Do NOT use any test names from the Issue Description - those are proposed NEW tests, not existing ones
3. Copy test names EXACTLY as they appear in the list

""",
            # Retry 3: final strict attempt
            """[FINAL ATTEMPT]
CRITICAL: Your previous responses contained invalid tests. This is your last attempt.

MANDATORY RULES:
1. Output ONLY test names that appear CHARACTER-FOR-CHARACTER in "Test Functions" section
2. The Issue Description may mention test names - these are PROPOSED tests that DO NOT EXIST in the codebase
3. You MUST select from the provided list
4. DO NOT GUESS or INFER test names - copy EXACTLY from the list

"""
        ]

        MAX_RETRIES = 3
        current_attempt = 1
        all_raw_outputs = [raw_output]
        all_trajs = [traj] if not isinstance(traj.get("response"), list) else [traj]

        # Initial validation.
        valid_tests, invalid_tests = validate_found_tests(found_tests)
        log_validation_details(found_tests, valid_tests, invalid_tests, current_attempt)

        # Retry loop.
        while should_retry(found_tests, invalid_tests) and current_attempt <= MAX_RETRIES:
            retry_prefix = RETRY_PREFIXES[current_attempt - 1]
            current_attempt += 1
            
            self.logger.warning(f"=== Triggering retry {current_attempt}/{MAX_RETRIES + 1} due to high invalid rate ===")
            self.logger.info(f"Invalid: {len(invalid_tests)}, Total: {len(found_tests)}, Threshold: {max(1, len(found_tests) // 2)}")

            # Build retry message with stronger constraints.
            retry_message = retry_prefix + message

            try:
                retry_model = make_model(
                    model=self.model_name,
                    backend=self.backend,
                    logger=self.logger,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    batch_size=1,
                )

                retry_traj = retry_model.codegen(retry_message, num_samples=1)[0]
                retry_traj["prompt"] = retry_message
                retry_raw_output = retry_traj["response"]

                all_raw_outputs.append(retry_raw_output)
                all_trajs.append(retry_traj)

                # Parse retry output.
                found_tests = self._parse_model_return_lines(retry_raw_output)
                
                # Re-validate retry output.
                valid_tests, invalid_tests = validate_found_tests(found_tests)
                log_validation_details(found_tests, valid_tests, invalid_tests, current_attempt)

                # Replace current output/traj with latest attempt.
                raw_output = retry_raw_output
                traj = retry_traj

            except Exception as e:
                self.logger.error(f"Retry {current_attempt} failed with error: {e}")
                break

        # Build final validated test records.
        valid_found_tests = []
        for test in valid_tests:
            valid_found_tests.append(
                {
                    "name": test,
                    "docstring": test_information_dict[test][0],
                    "line_number": test_information_dict[test][1]
                }
            )

        # Final logging.
        self.logger.info(f"=== Final Results after {current_attempt} attempt(s) ===")
        self.logger.info(f"Final raw output: {raw_output}")
        self.logger.info(f"Final valid tests: {[t['name'] for t in valid_found_tests]}")
        if invalid_tests:
            self.logger.warning(f"Remaining invalid tests (discarded): {invalid_tests}")

        # Metadata including retry history.
        metadata = {
            "raw_output_tests": raw_output,
            "found_tests": valid_found_tests,
            "retry_count": current_attempt - 1,
            "all_raw_outputs": all_raw_outputs,
            "final_invalid_tests": invalid_tests
        }

        return (
            valid_found_tests,
            metadata,
            traj,
        )


    def extract_coverage_elements(
        self,
        test_functions: List[str]
    ):
        """
        Export coverage elements touched by the selected test functions.
        Output includes `coverage_functions` and `coverage_classes`.
        """
        if getattr(self, '_use_lazy_loading', False):
            missing_tests = [t for t in test_functions 
                           if self.coverage_data.get(t) is None]
            if missing_tests:
                self.logger.info(f"Loading {len(missing_tests)} missing coverage entries")
                newly_loaded = self._load_coverage_for_tests(missing_tests)
                self.coverage_data.update(newly_loaded)
            loaded_coverage = {t: self.coverage_data.get(t) for t in test_functions}
        else:
            loaded_coverage = self.coverage_data
        
        coverage_elements = dict()

        for test_func in test_functions:
            if test_func not in loaded_coverage or loaded_coverage[test_func] is None:
                self.logger.warning(f"No coverage data found for test: {test_func}")
                continue
            
            coverage_elements[test_func] = loaded_coverage[test_func]

        return coverage_elements


    def _load_coverage_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Load coverage graph data from JSON file (unified format with nodes/edges)."""
        if not self.coverage_graph_path or not os.path.exists(self.coverage_graph_path):
            self.logger.warning(f"Coverage graph not found at: {self.coverage_graph_path}")
            return {}
        
        try:
            file_size_mb = os.path.getsize(self.coverage_graph_path) / (1024 * 1024)
            self.logger.info(f"Coverage graph file size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 500:
                self.logger.info(f"Large file detected, using lazy loading strategy")
                self._use_lazy_loading = True
                self._coverage_keys = self._load_coverage_keys_only()
                self.logger.info(f"Loaded {len(self._coverage_keys)} coverage keys (lazy mode)")
                return {key: None for key in self._coverage_keys}
            else:
                self._use_lazy_loading = False
                with open(self.coverage_graph_path, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"Loaded coverage graph with {len(data)} test entries")
                    sample_keys = list(data.keys())[:5]
                    self.logger.info(f"Sample coverage keys: {sample_keys}")
                    return data
        except Exception as e:
            self.logger.error(f"Error loading coverage graph: {e}")
            raise e


    def _load_coverage_keys_only(self) -> set:
        import ijson
        
        keys = set()
        with open(self.coverage_graph_path, 'rb') as f:
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if event == 'map_key' and prefix == '':
                    keys.add(value)
        return keys


    def _load_coverage_for_tests(self, test_names: List[str]) -> Dict[str, Any]:
        import ijson
        
        self.logger.info(f"Lazy loading coverage for {len(test_names)} tests...")
        
        result = {}
        test_names_set = set(test_names)
        
        with open(self.coverage_graph_path, 'rb') as f:
            parser = ijson.kvitems(f, '')
            for key, value in parser:
                if key in test_names_set:
                    result[key] = value
                    self.logger.debug(f"Loaded coverage for: {key}")
                    if len(result) == len(test_names_set):
                        break
        self.logger.info(f"Lazy loaded {len(result)} test coverage entries")
        return result


    def _load_domain_knowledge(self) -> Dict[str, List[str]]:
        """Load domain knowledge tokens for test functions."""
        if not self.domain_knowledge_path or not os.path.exists(self.domain_knowledge_path):
            self.logger.info(f"Domain knowledge not found at: {self.domain_knowledge_path}")
            return {}
        
        try:
            with open(self.domain_knowledge_path, 'r') as f:
                raw_data = json.load(f)
            
            # Convert list format to dict format
            # Input: [{"test_func": "path::name", "domain_knowledge_tokens": [...]}, ...]
            # Output: {"path::name": [...], ...}
            domain_map = {}
            for entry in raw_data:
                test_func = entry.get('test_func', '')
                tokens = entry.get('domain_knowledge_tokens', [])
                if test_func and tokens:  # Only store non-empty token lists
                    domain_map[test_func] = tokens
            
            self.logger.info(f"Loaded domain knowledge for {len(domain_map)} test functions from {self.domain_knowledge_path}")
            return domain_map
        except Exception as e:
            self.logger.error(f"Error loading domain knowledge: {e}")
            return {}


    def _merge_domain_knowledge(self):
        """Merge domain knowledge tokens into test_functions."""
        if not self.domain_knowledge:
            return
        
        merged_count = 0
        for test in self.test_functions:
            test_name = test.get('name', '')
            if test_name in self.domain_knowledge:
                test['domain_tokens'] = self.domain_knowledge[test_name]
                merged_count += 1
            else:
                test['domain_tokens'] = []
        
        self.logger.info(f"Merged domain knowledge into {merged_count}/{len(self.test_functions)} filtered test functions")


    def _clean_coverage_data(self, raw_coverage_data:dict) -> Dict[str, Dict[str, List[str]]]:
        deleted_cnt = 0
        cleaned_data = {}
        files = set([file[0] for file in self.files])
        classes = set([class_['name'] for class_ in self.classes])
        functions = set([func['name'] for func in self.functions])

        print(f"Starting cleaning coverage data... There are {len(raw_coverage_data)} test entries")
        for test_name, coverage_info in raw_coverage_data.items():
            cleaned_info = {
                'covered_functions': [],
                'covered_classes': []
            }
            pass

            if 'covered_functions' in coverage_info and coverage_info['covered_functions'] is not None:
                for cov_func in coverage_info['covered_functions']:
                    if cov_func.split('::')[0] not in files:
                        deleted_cnt += 1
                        continue

                    cov_func_name = cov_func.split('::')[1]
                    if '.' in cov_func_name:
                        cov_func_name = cov_func_name.split('.')[-1]
                    if (cov_func_name not in functions) and (cov_func_name not in classes):
                        deleted_cnt += 1
                        continue
                    cleaned_info['covered_functions'].append(cov_func)

            if 'covered_classes' in coverage_info and coverage_info['covered_classes'] is not None:
                for cov_class in coverage_info['covered_classes']:
                    if cov_class.split('::')[0] not in files:
                        deleted_cnt += 1
                        continue

                    cov_class_name = cov_class.split('::')[1]
                    if '.' in cov_class_name:
                        cov_class_name = cov_class_name.split('.')[0]
                    if (cov_class_name not in classes) and (cov_class_name not in functions):
                        deleted_cnt += 1
                        continue
                    cleaned_info['covered_classes'].append(cov_class)

            if cleaned_info['covered_functions'] or cleaned_info['covered_classes']:
                cleaned_data[test_name] = cleaned_info

        print(f"Completed cleaning coverage data. Found {len(cleaned_data)} valid test coverage entries.")
        print(f"Deleted {deleted_cnt} invalid coverage locations")
        return cleaned_data

    def _load_test_functions(self) -> List[Dict[str, str]]:
        """Load test functions with their metadata."""
        if not self.test_functions_path or not os.path.exists(self.test_functions_path):
            self.logger.warning(f"Test functions data not found at: {self.test_functions_path}")
            return []
        
        try:
            with open(self.test_functions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading test functions: {e}")
            return []


    def _discover_test_functions(self) -> List[Dict[str, str]]:
        """Discover test functions from the repository structure."""
        # If test functions are already loaded, return them
        if self.test_functions and len(self.test_functions) > 0:
            self.logger.info(f"Test functions already loaded: {len(self.test_functions)} functions")
            return self.test_functions
        
        self.logger.info("Starting to discover test functions from repository structure...")
        self.logger.info(f"Total files in structure: {len(self.files)}")
        
        test_functions = []
        test_files_scanned = 0
        
        for file_path, content in self.files:
            if 'test' in file_path.lower() and file_path.endswith('.py'):
                test_files_scanned += 1
                self.logger.debug(f"Scanning test file: {file_path}")
                
                # Use AST to parse and extract test functions
                extracted = self._extract_test_functions_from_file(file_path, content)
                self.logger.debug(f"  -> Extracted {len(extracted)} test functions from {file_path}")
                test_functions.extend(extracted)
        
        self.logger.info(f"Scanned {test_files_scanned} test files, found {len(test_functions)} test functions before cleaning")
        

        if test_functions:
            sample_funcs = [tf['name'] for tf in test_functions[:5]]
            self.logger.info(f"Sample discovered test functions: {sample_funcs}")
        
        cleaned = self._clean_test_functions(test_functions)
        self.logger.info(f"After cleaning: {len(cleaned)} test functions remain")
        
        return cleaned


    def _extract_test_functions_from_file(self, file_path: str, content) -> List[Dict]:
        """Extract test functions from a file, including class methods."""
        test_functions = []
        
        try:
            self.logger.debug(f"Processing {file_path}, content type: {type(content).__name__}")
            
            if isinstance(content, list):
                lines = content
                content = '\n'.join(content)
                self.logger.debug(f"  Content was list, converted to string with {len(lines)} lines")
            elif isinstance(content, str):
                lines = content.split('\n')  
                self.logger.debug(f"  Content is string with {len(lines)} lines")
            else:
                self.logger.warning(f"Unexpected content type for {file_path}: {type(content)}")
                return test_functions
            
            tree = ast.parse(content)
            self.logger.debug(f"  AST parsing successful for {file_path}")

            for node in ast.iter_child_nodes(tree):  
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_name = f"{file_path}::{node.name}"
                    
                    func_line = node.lineno - 1
                    try:
                        comments = self._extract_comments_near_function(lines, func_line)
                    except Exception as e:
                        comments = ""
                        self.logger.debug(f"Failed to extract comments for {test_name}: {e}")
                    
                    try:
                        func_source = ast.get_source_segment(content, node)
                        if func_source is None:
                            func_source = ""
                    except Exception:
                        func_source = ""
                    
                    test_functions.append({
                        'name': test_name,
                        'code': func_source,
                        'comments': comments,
                        'docstring': ast.get_docstring(node) or "",
                        'line_number': node.lineno,            
                        'file_path': file_path,
                        'class_name': None,
                        'method_name': node.name
                    })
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                            format1 = f"{file_path}::{class_name}::{item.name}"  # Class::method
                            format2 = f"{file_path}::{class_name}.{item.name}"   # Class.method
                            
                            if format1 in self.coverage_data:
                                test_name = format1
                            elif format2 in self.coverage_data:
                                test_name = format2
                            else:
                                test_name = format1
                            
                            func_line = item.lineno - 1
                            try:
                                comments = self._extract_comments_near_function(lines, func_line)
                            except Exception as e:
                                comments = ""
                                self.logger.debug(f"Failed to extract comments for {test_name}: {e}")
                            
                            try:
                                func_source = ast.get_source_segment(content, item)
                                if func_source is None:
                                    func_source = ""
                            except Exception:
                                func_source = ""

                            test_functions.append({
                                'name': test_name,
                                'code': func_source,
                                'comments': comments,
                                'docstring': ast.get_docstring(item) or "", 
                                'line_number': item.lineno,              
                                'file_path': file_path,
                                'class_name': class_name,
                                'method_name': item.name
                            })
            
            self.logger.debug(f"  Found {len(test_functions)} test functions in {file_path}")
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error parsing {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        return test_functions


    def _extract_test_functions_comprehensive(self, file_path: str, content: Union[str, List[str]]) -> List[Dict[str, str]]:
        """Extract test functions including those in test classes."""
        test_functions = []
        
        try:
            # Handle case where content might be a list of lines instead of a string
            if isinstance(content, list):
                lines = content
                content = '\n'.join(content)
            elif not isinstance(content, str):
                print(f"Warning: Unexpected content type for {file_path}: {type(content)}")
                return test_functions
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    # Get the full qualified name (including class if applicable)
                    full_name = self._get_full_function_name(node, tree)
                    docstring = ast.get_docstring(node) or ""
                    
                    # Try to extract comments near the function
                    func_line = node.lineno - 1  # Convert to 0-based
                    comments = self._extract_comments_near_function(lines, func_line)

                    # Use docstring if available, otherwise use comments
                    documentation = docstring if docstring else comments

                    test_functions.append({
                        'name': f"{file_path}::{full_name}",
                        'docstring': documentation,
                        'line_number': node.lineno
                    })
                    
        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path} due to syntax error: {e}")
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
        
        return test_functions


    def _extract_comments_near_function(self, lines: List[str], func_line: int) -> str:
        """Extract comments before or after the function definition."""
        comments = []
        
        # Check lines before the function (up to 3 lines)
        for i in range(max(0, func_line - 3), func_line):
            line = lines[i].strip()
            if line.startswith('#'):
                comments.append(line[1:].strip())  # Remove # and whitespace
        
        # Check the first few lines inside the function for comments
        for i in range(func_line + 1, min(len(lines), func_line + 5)):
            line = lines[i].strip()
            if line.startswith('#'):
                comments.append(line[1:].strip())
            elif line and not line.startswith('"""') and not line.startswith("'''"):
                break  # Stop if we hit non-comment, non-docstring code
        
        return ' | '.join(comments) if comments else ""


    def _get_full_function_name(self, func_node: ast.FunctionDef, tree: ast.AST) -> str:
        """Get the full qualified name of a function (including class name if applicable)."""
        # Find if this function is inside a class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is func_node:
                        return f"{node.name}.{func_node.name}"
        
        return func_node.name


    def _clean_test_functions(self, raw_test_functions) -> List[Dict[str, str]]:
        
        self.logger.info(f"Cleaning test functions: {len(raw_test_functions)} raw functions")
        self.logger.info(f"Coverage data has {len(self.coverage_data)} test entries")
        
        clean_test_functions = []
        valid_test_names = set(self.coverage_data.keys())
        
        sample_coverage_keys = list(valid_test_names)[:5]
        self.logger.info(f"Sample coverage_data keys: {sample_coverage_keys}")
        
        if raw_test_functions:
            sample_raw_names = [tf['name'] for tf in raw_test_functions[:5]]
            self.logger.info(f"Sample raw test function names: {sample_raw_names}")
        
        not_found_count = 0
        normalized_match_count = 0  
        
        for test_func in raw_test_functions:
            original_name = test_func['name']
            
            if original_name in valid_test_names:
                clean_test_functions.append(test_func)
                continue
            
            normalized_name = self._try_normalize_test_name(original_name)
            if normalized_name and normalized_name in valid_test_names:
                test_func['name'] = normalized_name
                clean_test_functions.append(test_func)
                normalized_match_count += 1
                self.logger.debug(f"Fallback match: {original_name} -> {normalized_name}")
                continue
            
            not_found_count += 1

            if not_found_count <= 10:
                self.logger.debug(f"Test function not in coverage data: {original_name}")
        
        if not_found_count > 10:
            self.logger.info(f"... and {not_found_count - 10} more test functions not in coverage data")
        
        if normalized_match_count > 0:
            self.logger.info(f"Fallback normalization matched {normalized_match_count} additional tests")
        
        self.logger.info(f"Cleaning complete: {len(clean_test_functions)} functions kept, {not_found_count} filtered out")
        

        ast_discovered_names = set(tf['name'] for tf in raw_test_functions)

        in_coverage_not_in_ast = valid_test_names - ast_discovered_names

        self.logger.info(f"=== Coverage vs AST Mismatch Analysis ===")
        self.logger.info(f"Tests in coverage: {len(valid_test_names)}")
        self.logger.info(f"Tests discovered by AST: {len(ast_discovered_names)}")
        self.logger.info(f"Tests in BOTH: {len(valid_test_names & ast_discovered_names)}")
        self.logger.info(f"Tests in coverage but NOT in AST: {len(in_coverage_not_in_ast)}")
        self.logger.info(f"Tests in AST but NOT in coverage: {len(ast_discovered_names - valid_test_names)}")

        if in_coverage_not_in_ast:
            self.logger.warning(f"=== Tests in coverage but NOT found by AST ({len(in_coverage_not_in_ast)} total) ===")
            
            tests_by_file = {}
            for test_name in sorted(in_coverage_not_in_ast):
                # test_name: "path/to/file.py::test_function"
                if '::' in test_name:
                    file_path, func_name = test_name.rsplit('::', 1)
                else:
                    file_path, func_name = "unknown", test_name
                
                if file_path not in tests_by_file:
                    tests_by_file[file_path] = []
                tests_by_file[file_path].append(func_name)
            
            for file_path in sorted(tests_by_file.keys()):
                funcs = tests_by_file[file_path]
                self.logger.warning(f"  File: {file_path} ({len(funcs)} missing tests)")
                for func_name in funcs[:20]:  
                    self.logger.warning(f"    - {func_name}")
                if len(funcs) > 20:
                    self.logger.warning(f"    ... and {len(funcs) - 20} more")


        return clean_test_functions


    def _try_normalize_test_name(self, test_name: str) -> Optional[str]:
        prefix = f"{self.instance_id}/"
        
        if test_name.startswith(prefix):
            return test_name[len(prefix):]
        
        return None



    def _diagnose_coverage_mismatch(self):
        self.logger.info("=== Starting coverage mismatch diagnosis ===")
        
        all_ast_test_names = set()
        for file_path, content in self.files:
            if 'test' in file_path.lower() and file_path.endswith('.py'):
                extracted = self._extract_test_functions_from_file(file_path, content)
                for tf in extracted:
                    all_ast_test_names.add(tf['name'])
        
        coverage_test_names = set(self.coverage_data.keys())
        
        in_coverage_not_in_ast = coverage_test_names - all_ast_test_names

        in_ast_not_in_coverage = all_ast_test_names - coverage_test_names

        in_both = coverage_test_names & all_ast_test_names
        
        self.logger.info(f"Coverage test names: {len(coverage_test_names)}")
        self.logger.info(f"AST discovered test names: {len(all_ast_test_names)}")
        self.logger.info(f"In both: {len(in_both)}")
        self.logger.info(f"In coverage but NOT in AST: {len(in_coverage_not_in_ast)}")
        self.logger.info(f"In AST but NOT in coverage: {len(in_ast_not_in_coverage)}")
        

        if in_coverage_not_in_ast:
            samples = list(in_coverage_not_in_ast)[:10]
            self.logger.warning(f"Sample tests in coverage but NOT found by AST: {samples}")
            
            for sample in samples[:3]:
                if '::' in sample:
                    file_path = sample.split('::')[0]
                    file_exists = any(fp == file_path for fp, _ in self.files)
                    self.logger.warning(f"  {sample} -> file exists in structure: {file_exists}")
        
        self.logger.info("=== Coverage mismatch diagnosis complete ===")


    def _format_test_functions_for_prompt(self, test_functions: List[Dict[str, str]]) -> str:
        """Format test functions for LLM prompt."""
        formatted = []
        for test_func in test_functions:
            parts = [test_func['name']]
            if test_func.get('docstring'):
                parts.append(f"  Description: {test_func['docstring']}")
            if test_func.get('domain_tokens'):
                parts.append(f"  Related Concepts: {', '.join(test_func['domain_tokens'])}")
            formatted.append('\n'.join(parts))
        return '\n\n'.join(formatted)


    def _parse_model_return_lines(self, content: str) -> List[str]:
        """Parse model return lines from code blocks."""
        if not content:
            return []
        
        # Extract content from code blocks
        code_blocks = extract_code_blocks(content)
        if code_blocks:
            return code_blocks[0].strip().split('\n')
        return content.strip().split('\n')



class BlindSpotAnalyzer(BaseLocalizer):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        found_tests,
        coverage_elements,
        domain_knowledge_path: Optional[str] = None
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 3000
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.found_tests = found_tests
        self.coverage_elements = coverage_elements
        self.analyze_blind_spots_prompt = analyze_blind_spots_prompt
        self.localize_suspicious_locations_prompt = localize_suspicious_locations_prompt
        
        # Load domain knowledge if path provided
        self.domain_knowledge = self._load_domain_knowledge(domain_knowledge_path, instance_id)
        if self.domain_knowledge:
            self.logger.info(f"Domain knowledge loaded for BlindSpotAnalyzer: {len(self.domain_knowledge)} entries")


    def _load_domain_knowledge(self, domain_knowledge_path: Optional[str], instance_id: str) -> Dict[str, List[str]]:
        """Load domain knowledge tokens for test functions."""
        if not domain_knowledge_path:
            return {}
        
        file_path = os.path.join(domain_knowledge_path, instance_id + ".json")
        
        if not os.path.exists(file_path):
            self.logger.info(f"Domain knowledge not found at: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            
            # Convert list format to dict format
            domain_map = {}
            for entry in raw_data:
                test_func = entry.get('test_func', '')
                tokens = entry.get('domain_knowledge_tokens', [])
                if test_func and tokens:
                    domain_map[test_func] = tokens
            
            self.logger.info(f"Loaded domain knowledge from {file_path}: {len(domain_map)} entries with tokens")
            return domain_map
        except Exception as e:
            self.logger.error(f"Error loading domain knowledge: {e}")
            return {}


    def analyze_blind_spots(
        self,
        mock
    ):
        """
        Analyze why passing tests failed to detect the reported issue.
        """
        from util.api_requests import num_tokens_from_messages
        from util.model import make_model

        if not self.found_tests:
            return {}, {"raw_output_blind_spots": ""}, {}

        # Build test details (no truncation at first).
        test_details_info = self._build_test_details(self.found_tests)

        if not test_details_info:
            self.logger.warning("No test details available for blind spots analysis")
            return {}, {"raw_output_blind_spots": ""}, {}

        # Build analysis prompt.
        message = self.analyze_blind_spots_prompt.format(
            problem_statement=self.problem_statement,
            test_functions='\n\n'.join(test_details_info)
        ).strip()

        self.logger.info(f"Analyzing test blind spots with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return {}, {"raw_output_blind_spots": ""}, traj

        # Context-length guard.
        def message_too_long(msg):
            return num_tokens_from_messages(msg, self.model_name) >= MAX_CONTEXT_LENGTH

        # Track currently used truncation settings.
        current_tests = list(self.found_tests)
        current_max_code_lines = None
        current_max_edges = None

        # Stage 1: truncate coverage edges first.
        edge_limits = [300, 200, 150, 100, 70, 50, 35, 25, 15]
        for max_edges in edge_limits:
            if not message_too_long(message):
                break
            current_max_edges = max_edges
            test_details_info = self._build_test_details(
                current_tests, 
                max_code_lines=current_max_code_lines, 
                max_edges=current_max_edges
            )
            message = self.analyze_blind_spots_prompt.format(
                problem_statement=self.problem_statement,
                test_functions='\n\n'.join(test_details_info)
            ).strip()
            self.logger.info(f"[Stage 1] Limiting edges to {max_edges} per test due to length limit")

        # Stage 2: truncate source code lines.
        code_line_limits = [50, 30, 20, 10]
        for max_lines in code_line_limits:
            if not message_too_long(message):
                break
            current_max_code_lines = max_lines
            test_details_info = self._build_test_details(
                current_tests, 
                max_code_lines=current_max_code_lines, 
                max_edges=current_max_edges
            )
            message = self.analyze_blind_spots_prompt.format(
                problem_statement=self.problem_statement,
                test_functions='\n\n'.join(test_details_info)
            ).strip()
            self.logger.info(f"[Stage 2] Limiting code to {max_lines} lines per test due to length limit")

        # Stage 3: reduce number of tests.
        while message_too_long(message) and len(current_tests) > 1:
            current_tests = current_tests[:-1]
            test_details_info = self._build_test_details(
                current_tests, 
                max_code_lines=current_max_code_lines, 
                max_edges=current_max_edges
            )
            message = self.analyze_blind_spots_prompt.format(
                problem_statement=self.problem_statement,
                test_functions='\n\n'.join(test_details_info)
            ).strip()
            self.logger.info(f"[Stage 3] Reducing to {len(current_tests)} tests due to length limit")

        # Stage 4 (fallback): extreme truncation for single-test case.
        if message_too_long(message) and len(current_tests) == 1:
            # Further compress single-test context.
            extreme_edge_limits = [10, 5]
            extreme_code_limits = [5, 3]
            
            for max_edges, max_lines in zip(extreme_edge_limits, extreme_code_limits):
                if not message_too_long(message):
                    break
                current_max_edges = max_edges
                current_max_code_lines = max_lines
                test_details_info = self._build_test_details(
                    current_tests, 
                    max_code_lines=current_max_code_lines, 
                    max_edges=current_max_edges
                )
                message = self.analyze_blind_spots_prompt.format(
                    problem_statement=self.problem_statement,
                    test_functions='\n\n'.join(test_details_info)
                ).strip()
                self.logger.info(f"[Stage 4] Extreme truncation: max_code_lines={max_lines}, max_edges={max_edges}")


        # Final safety check after truncation.
        if message_too_long(message):
            self.logger.warning(
                f"Single test still exceeds context limit after all truncation stages. "
                f"Test: {current_tests[0]['name'] if current_tests else 'None'}, "
                f"max_code_lines={current_max_code_lines}, max_edges={current_max_edges}"
            )
            # Last-resort truncation.
            test_details_info = self._build_test_details(
                current_tests, 
                max_code_lines=5, 
                max_edges=10
            )
            message = self.analyze_blind_spots_prompt.format(
                problem_statement=self.problem_statement,
                test_functions='\n\n'.join(test_details_info)
            ).strip()
            self.logger.info(f"[Stage 5] Extreme truncation: max_code_lines=5, max_edges=10")
            
            if message_too_long(message):
                self.logger.error("Message still too long after extreme truncation, skipping this analysis")
                return {}, {"raw_output_blind_spots": "Error: Message too long even after extreme truncation"}, {}
        

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        

        # Keep optional debug hook for raw output.
        # with open('debug_raw_output.txt', 'w') as f:
        #     f.write(raw_output)
        # raise ValueError("Debugging raw_output")

        # Parse blind spot output.
        blind_spots_analysis = self._parse_blind_spots_analysis(raw_output)        


        self.logger.info(f"Blind spots analysis raw output: {raw_output}")
        self.logger.info(f"Parsed blind spots analysis: {blind_spots_analysis}")
        
        return (
            blind_spots_analysis,
            {"raw_output_blind_spots": raw_output},
            traj
        )


    def localize_suspicious_locations(
        self,
        blind_spots_analysis: Dict[str, Any],
        mock=False
    ):
        """
        Map blind-spot analysis to suspicious code locations.
        """

        from util.api_requests import num_tokens_from_messages
        from util.model import make_model
        
        if not blind_spots_analysis or not self.found_tests:
            return [], {"raw_output_mapping": ""}, {}

        # Collect all covered code nodes from selected tests.
        all_coverage_locations = []
        for test in self.found_tests:
            test_func = test['name']
            if test_func in self.coverage_elements:
                coverage = self.coverage_elements[test_func]
                all_coverage_locations.extend(coverage.get("nodes", []))

        # Deduplicate candidates.
        unique_locations = list(set(all_coverage_locations))
        
        # Filter out test-side locations.
        from util.test_filter import filter_test_locations
        unique_locations = filter_test_locations(unique_locations)
        self.logger.info(f"After filtering test locations: {len(unique_locations)} locations remain")
        
        traceback_locations = extract_and_validate_locations_from_traceback(
            self.problem_statement,
            self.files,
            self.classes,
            self.functions,
            logger=self.logger
        )
        self.logger.info(f"Extracted {len(traceback_locations)} valid locations from traceback: {traceback_locations}")
        
        # Merge coverage and traceback candidates.
        extended_candidates = list(set(unique_locations + traceback_locations))
        self.logger.info(f"Extended candidate pool: {len(extended_candidates)} total locations")
        
        # Extract location fragments from blind-spot text.
        mentioned_fragments = self._extract_mentioned_locations_from_blind_spots(
            blind_spots_analysis
        )
        
        if mentioned_fragments:
            # Score each candidate against mentioned fragments.
            location_scores = []
            for loc in unique_locations:
                score = self._calculate_match_score(loc, mentioned_fragments)
                location_scores.append((loc, score))
            
            # Sort by score (higher is better).
            location_scores.sort(key=lambda x: x[1], reverse=True)
            unique_locations = [loc for loc, score in location_scores]
            
            # Count prioritized candidates (score > 0).
            priority_count = sum(1 for _, score in location_scores if score > 0)
            if priority_count > 0:
                self.logger.info(f"Reordered coverage locations: {priority_count} matched blind_spots mentions, "
                                f"{len(unique_locations) - priority_count} others")

        coverage_locations_text = '\n'.join([f"- {loc}" for loc in unique_locations])
        
        if traceback_locations:
            coverage_locations_text += "\n\n**Additional locations from traceback (verified in codebase, high priority):**\n"
            coverage_locations_text += '\n'.join([f"- {loc}" for loc in traceback_locations])


        blind_spots_text = blind_spots_analysis

        # Build mapping prompt.
        message = self.localize_suspicious_locations_prompt.format(
            problem_statement=self.problem_statement,
            blind_spots_analysis=blind_spots_text,
            test_coverage_locations=coverage_locations_text
        ).strip()

        self.logger.info(f"Mapping blind spots to code with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_mapping": ""}, traj

        # Context-length guard.
        def message_too_long(msg):
            return num_tokens_from_messages(msg, self.model_name) >= MAX_CONTEXT_LENGTH
        
        # Progressive truncation for oversized prompts.
        truncation_round = 0
        while message_too_long(message) and len(unique_locations) > 5:
            truncation_round += 1
            # Aggressive shrink: keep 70% each round.
            new_size = max(5, int(len(unique_locations) * 0.7))
            unique_locations = unique_locations[:new_size]
            coverage_locations_text = '\n'.join([f"- {loc}" for loc in unique_locations])
            if traceback_locations:
                coverage_locations_text += "\n\n**Additional locations from traceback (verified in codebase, high priority):**\n"
                coverage_locations_text += '\n'.join([f"- {loc}" for loc in traceback_locations])
            message = self.localize_suspicious_locations_prompt.format(
                problem_statement=self.problem_statement,
                blind_spots_analysis=blind_spots_text,
                test_coverage_locations=coverage_locations_text
            ).strip()
            self.logger.info(f"[Truncation round {truncation_round}] Reducing coverage locations to {len(unique_locations)} due to length limit")
        
        # Final truncation fallback.
        if message_too_long(message):
            self.logger.warning(f"Message still too long with {len(unique_locations)} locations, attempting minimal set")
            # Extreme fallback: keep top 5 locations only.
            unique_locations = unique_locations[:5]
            coverage_locations_text = '\n'.join([f"- {loc}" for loc in unique_locations])
            if traceback_locations:
                coverage_locations_text += "\n\n**Additional locations from traceback (verified in codebase, high priority):**\n"
                coverage_locations_text += '\n'.join([f"- {loc}" for loc in traceback_locations])
            message = self.localize_suspicious_locations_prompt.format(
                problem_statement=self.problem_statement,
                blind_spots_analysis=blind_spots_text,
                test_coverage_locations=coverage_locations_text
            ).strip()
            
            if message_too_long(message):
                self.logger.error("Message still too long after extreme truncation, returning empty results")
                return [], {"raw_output_mapping": "Error: Message too long"}, {}
        
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        # Parse mapping output.
        suspicious_locations = self._parse_model_return_lines(raw_output)
        suspicious_locations = self._normalize_init_methods(suspicious_locations)

        def validate_single_location(loc: str) -> Tuple[bool, Optional[str]]:
            """Validate one location and return normalized form if valid."""
            if not loc or '::' not in loc:
                return False, None
            
            # Prefer exact matches from candidate pool.
            if loc in extended_candidates:
                return True, loc
            
            # Fallback to AST-based validation for unseen candidates.
            is_valid, normalized = validate_location_in_project(
                loc, self.files, self.classes, self.functions
            )
            if is_valid:
                return True, normalized or loc
            
            return False, None

        def validate_all_locations(locations: List[str]) -> Tuple[List[str], List[str]]:
            """Validate all locations and split valid/invalid ones."""
            valid = []
            invalid = []
            for loc in locations:
                is_valid, normalized = validate_single_location(loc)
                if is_valid and normalized:
                    valid.append(normalized)
                else:
                    invalid.append(loc)
            return valid, invalid

        def log_validation_details(locations, valid, invalid, attempt_num):
            """Log location validation details."""
            self.logger.info(f"=== Validation Details (Attempt {attempt_num}) ===")
            self.logger.info(f"LLM returned {len(locations)} location(s)")
            self.logger.info(f"Valid locations ({len(valid)}): {valid}")
            if invalid:
                self.logger.warning(f"Invalid locations ({len(invalid)}): {invalid}")

        # Initial validation.
        MAX_RETRIES = 3
        current_attempt = 1
        all_raw_outputs = [raw_output]
        
        valid_locations, invalid_locations = validate_all_locations(suspicious_locations)
        log_validation_details(suspicious_locations, valid_locations, invalid_locations, current_attempt)
        
        # Retry loop.
        while should_retry_suspicious_locations(suspicious_locations, valid_locations) and current_attempt <= MAX_RETRIES:
            retry_prefix = SUSPICIOUS_LOCATION_RETRY_PREFIXES[current_attempt - 1]
            current_attempt += 1
            
            self.logger.warning(f"=== Triggering retry {current_attempt}/{MAX_RETRIES + 1} due to high invalid rate ===")
            self.logger.info(f"Invalid: {len(invalid_locations)}, Total: {len(suspicious_locations)}")
            
            # Build explicit valid-candidate list.
            valid_candidate_list = '\n'.join([f"- {loc}" for loc in extended_candidates])
            
            # Build constrained retry prompt.
            retry_message = retry_prefix.format(valid_candidate_list=valid_candidate_list) + message
            
            try:
                retry_traj = model.codegen(retry_message, num_samples=1)[0]
                retry_traj["prompt"] = retry_message
                retry_raw_output = retry_traj["response"]
                
                all_raw_outputs.append(retry_raw_output)
                
                # Parse retry output.
                suspicious_locations = self._parse_model_return_lines(retry_raw_output)
                suspicious_locations = self._normalize_init_methods(suspicious_locations)
                
                # Re-validate retry output.
                valid_locations, invalid_locations = validate_all_locations(suspicious_locations)
                log_validation_details(suspicious_locations, valid_locations, invalid_locations, current_attempt)
                
                # Keep latest output/traj as final.
                raw_output = retry_raw_output
                traj = retry_traj
                
            except Exception as e:
                self.logger.error(f"Retry {current_attempt} failed with error: {e}")
                break
        
        # Deduplicate while preserving order.
        valid_locations = list(dict.fromkeys(valid_locations))
        
        # Second-pass filter for test-side locations.
        from util.test_filter import filter_test_locations
        before_filter = len(valid_locations)
        valid_locations = filter_test_locations(valid_locations)
        if before_filter != len(valid_locations):
            self.logger.info(f"Filtered out {before_filter - len(valid_locations)} test locations from final output")

        # Trigger fallback only when no valid location remains.
        should_fallback = (len(valid_locations) == 0)
        
        if should_fallback:
            self.logger.info("Coverage-based analysis yielded no valid results, triggering Issue-Direct fallback")
            
            fallback_locations = extract_code_references_from_issue_description(
                self.problem_statement,
                self.files,
                self.classes,
                self.functions,
                logger=self.logger
            )
            
            if fallback_locations:
                # Filter out test locations from fallback output.
                fallback_locations = filter_test_locations(fallback_locations)
                
                if fallback_locations:
                    valid_locations = fallback_locations
                    self.logger.info(f"Issue-Direct fallback succeeded: {len(fallback_locations)} locations extracted")
                    self.logger.info(f"Fallback locations: {fallback_locations}")
                else:
                    self.logger.warning("Issue-Direct fallback locations were all test files, keeping empty result")
            else:
                self.logger.warning("Issue-Direct fallback found no valid locations")
        elif "NOT_IN_COVERAGE" in raw_output:
            # If valid locations exist, keep them even if output mentions NOT_IN_COVERAGE.
            self.logger.warning(f"Model output contains NOT_IN_COVERAGE note, but {len(valid_locations)} valid locations were found. Using valid locations.")

        # Final logging.
        self.logger.info(f"=== Final Results after {current_attempt} attempt(s) ===")
        self.logger.info(f"Final raw output: {raw_output}")
        self.logger.info(f"Final valid locations: {valid_locations}")
        if invalid_locations:
            self.logger.warning(f"Remaining invalid locations (discarded): {invalid_locations}")

        return (
            valid_locations,
            {
                "raw_output_mapping": raw_output, 
                "mapped_locations": valid_locations,
                "retry_count": current_attempt - 1,
                "all_raw_outputs": all_raw_outputs
            },
            traj
        )


    def _normalize_init_methods(self, locations: List[str]) -> List[str]:
        """
        Normalize location list: convert __init__ methods to class level.
        
        This is because modifications to __init__ typically involve:
        1. Parameter descriptions in class docstrings
        2. __init__ method signature
        3. Parent class __init__ calls
        
        These modifications are best annotated at the class level.
        
        Args:
            locations: Original location list
            
        Returns:
            Normalized location list (deduplicated)
        """
        normalized = []
        for loc in locations:
            if '.__init__' in loc:
                normalized_loc = loc.replace('.__init__', '')
                self.logger.debug(f"Normalized {loc} -> {normalized_loc}")
                normalized.append(normalized_loc)
            else:
                normalized.append(loc)
        
        seen = set()
        result = []
        for loc in normalized:
            if loc not in seen:
                seen.add(loc)
                result.append(loc)
        
        return result


    # Coverage graph formatting helper.
    def _format_coverage_graph_for_prompt(self, test_name: str, coverage_data: Dict[str, List[str]], max_edges: int = None, use_grouped_format: bool = True) -> str:
        """Format coverage graph data for LLM prompt.
        
        Args:
            test_name: Name of the test function
            coverage_data: Coverage data dictionary containing 'edges'
            max_edges: Maximum number of edges to include (None for no limit)
            use_grouped_format: If True, use grouped format (caller → [callees]); 
                            if False, use original line-by-line format
        """
        call_relations = coverage_data.get('edges', [])
        
        if not call_relations:
            return "No call relations available"
        
        total_edges = len(call_relations)
        
        # Prefer grouped format by default.
        if use_grouped_format:
            return self._format_coverage_grouped(call_relations, max_edges, total_edges)
        else:
            # Original flat format (backward-compatible).
            if max_edges is not None and total_edges > max_edges:
                call_relations = call_relations[:max_edges]
                truncated_note = f"\n... (truncated, showing {max_edges}/{total_edges} edges)"
            else:
                truncated_note = ""
            
            explain = "Call relations (caller -> callee):"
            relations_text = '\n'.join(call_relations)
            
            return f"{explain}\n{relations_text}{truncated_note}"


    def _format_coverage_grouped(self, call_relations: List[str], max_edges: int = None, total_edges: int = None) -> str:
        """
        Group and format coverage edges for compact prompt representation.

        Format:
        Caller →
        • callee1
        • callee2
        
        Args:
            call_relations: list like ["caller -> callee", ...]
            max_edges: max number of edges to include
            total_edges: original total edge count
        """
        from collections import defaultdict
        
        # Parse and group by caller.
        caller_to_callees = defaultdict(list)
        for edge in call_relations:
            if ' -> ' in edge:
                parts = edge.split(' -> ', 1)
                if len(parts) == 2:
                    caller, callee = parts
                    caller_to_callees[caller.strip()].append(callee.strip())
        
        # Build output text.
        lines = ["Coverage Information (format: Caller → [Callees]):"]
        lines.append("")
        
        edge_count = 0
        total_expected = total_edges if total_edges else sum(len(v) for v in caller_to_callees.values())
        
        for caller, callees in caller_to_callees.items():
            # Stop if edge budget is reached.
            if max_edges is not None and edge_count >= max_edges:
                remaining = total_expected - edge_count
                lines.append(f"... (truncated, showing {edge_count}/{total_expected} edges, {remaining} more not shown)")
                break
            
            lines.append(f"{caller} →")
            
            # Add callees under current caller with edge limit.
            available_slots = max_edges - edge_count if max_edges else len(callees)
            displayed_callees = callees[:available_slots]
            
            for callee in displayed_callees:
                lines.append(f"  • {callee}")
                edge_count += 1
                
                if max_edges is not None and edge_count >= max_edges:
                    break
            
            # Mark omitted callees for this caller if truncated.
            if len(callees) > len(displayed_callees):
                hidden = len(callees) - len(displayed_callees)
                lines.append(f"  ... and {hidden} more callees")
            
            lines.append("")  # blank separator
        
        return '\n'.join(lines)


    def _build_test_details(
        self,
        tests: List[Dict],
        max_code_lines: int = None,
        max_edges: int = None
    ) -> List[str]:
        """
        Build test detail blocks for prompt construction.
        
        Args:
            tests: selected test entries
            max_code_lines: max lines of test source code (None = no limit)
            max_edges: max coverage edges per test (None = no limit)
            
        Returns:
            test_details: formatted text blocks
        """
        test_details = []
        
        for test in tests:
            test_func = test['name']
            test_info_parts = []

            # 1) basic metadata
            test_info_parts.append(f"=== Test Function: {test_func} ===")

            # 2) test source code
            test_code = self._extract_test_function_code(test_func)
            if test_code:
                test_info_parts.append("Test Code:")
                test_info_parts.append("```python")
                
                # Optional source truncation.
                if max_code_lines is not None:
                    code_lines = test_code.split('\n')
                    if len(code_lines) > max_code_lines:
                        test_code = '\n'.join(code_lines[:max_code_lines])
                        test_code += f"\n    # ... (code truncated, showing {max_code_lines}/{len(code_lines)} lines)"
                
                test_info_parts.append(test_code)
                test_info_parts.append("```")
            else:
                test_info_parts.append("Test Code: [Code not found]")

            # 3) coverage information
            if test_func in self.coverage_elements:
                coverage = self.coverage_elements[test_func]
                # Use grouped format to reduce prompt size.
                coverage_text = self._format_coverage_graph_for_prompt(
                    test_func, 
                    coverage, 
                    max_edges=max_edges,
                )
                test_info_parts.append("Coverage Information:")
                test_info_parts.append(coverage_text)
            else:
                test_info_parts.append("Coverage Information: [No coverage data]")

            test_func_info = next((tf for tf in self.found_tests if tf['name'] == test_func), None)
            if test_func_info and test_func_info.get('docstring'):
                test_info_parts.append("Test Description:")
                test_info_parts.append(test_func_info['docstring'])
            
            if self.domain_knowledge and test_func in self.domain_knowledge:
                domain_tokens = self.domain_knowledge[test_func]
                test_info_parts.append("Related Concepts (from historical analysis):")
                test_info_parts.append(f"  {', '.join(domain_tokens)}")
            
            test_details.append('\n'.join(test_info_parts))
        
        return test_details




    def _extract_test_function_code(self, test_function_name: str) -> str:
        """
        Extract source code for a test function/method from repository files.
        
        Args:
            test_function_name: supported formats:
                - "file_path::function_name" (top-level function)
                - "file_path::ClassName.method_name" (class method)
                - "file_path::ClassName::method_name" (class method)
            
        Returns:
            test_code: extracted source, empty string if not found
        """
        if '::' not in test_function_name:
            return ""
        
        # Unified parsing for multiple name formats.
        parts = test_function_name.split('::')
        if len(parts) < 2:
            return ""
        
        file_path = parts[0]
        
        if len(parts) == 2:
            # Format: file::identifier or file::ClassName.method
            identifier = parts[1]
            if '.' in identifier:
                class_name, method_name = identifier.split('.', 1)
            else:
                class_name = None
                method_name = identifier
        elif len(parts) == 3:
            # Format: file::ClassName::method_name
            class_name = parts[1]
            method_name = parts[2]
        else:
            # Unknown format (e.g., nested structures): fail safely.
            self.logger.warning(f"Unsupported test function name format: {test_function_name}")
            return ""
        
        # Locate file content.
        file_content = None
        for fp, content in self.files:
            if fp == file_path:
                file_content = content
                break
        
        if not file_content:
            self.logger.warning(f"File {file_path} not found for test function {test_function_name}")
            return ""
        
        try:
            # Normalize file content into string + line list.
            if isinstance(file_content, list):
                lines = file_content
                file_content_str = '\n'.join(file_content)
            else:
                file_content_str = file_content
                lines = file_content.split('\n')
            
            # Parse AST.
            tree = ast.parse(file_content_str)
            
            # Find target function node.
            target_node = None
            target_class_node = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
                    target_class_node = node
                    # Search methods inside matched class.
                    for child_node in ast.walk(node):
                        if (isinstance(child_node, ast.FunctionDef) and 
                            child_node.name == method_name and 
                            child_node != node):  # avoid matching class node itself
                            target_node = child_node
                            break
                elif isinstance(node, ast.FunctionDef) and not class_name and node.name == method_name:
                    target_node = node
                    break
            
            if not target_node:
                self.logger.warning(f"Function {func_name} not found in {file_path}")
                return ""
            
            # Extract line range.
            start_line = target_node.lineno - 1  # AST is 1-based; list is 0-based
            
            # Prefer native end_lineno when available.
            if hasattr(target_node, 'end_lineno') and target_node.end_lineno:
                end_line = target_node.end_lineno
            else:
                # Fallback end-line estimation for older Python versions.
                end_line = start_line + 1
                indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
                
                for i in range(start_line + 1, len(lines)):
                    line = lines[i]
                    if line.strip() == "":  # skip blank lines
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    end_line = i + 1
            
            # Return extracted source.
            function_lines = lines[start_line:end_line]
            function_code = '\n'.join(function_lines)
            
            return function_code
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error parsing {file_path}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting test function code: {e}")
            return ""



    def _parse_blind_spots_analysis(self, raw_output: str) -> Dict[str, Any]:
        match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
        blind_spots = match.group(1).strip() if match else ""
        return blind_spots

    def _parse_model_return_lines(self, content: str) -> List[str]:
        """Parse model return lines from code blocks."""
        if not content:
            return []
        
        # Extract content from code blocks
        code_blocks = extract_code_blocks(content)
        raw_lines = code_blocks[0].strip().split('\n') if code_blocks else content.strip().split('\n')
        
        import re
        cleaned = []
        for line in raw_lines:
            clean = re.sub(r'^[\s\-\*\(\)0-9.]+', '', line)
            clean = re.sub(r'\s*\[.*$', '', clean).strip()
            if clean and '::' in clean:
                cleaned.append(clean)
        
        return cleaned



    def _extract_mentioned_locations_from_blind_spots(self, blind_spots_text: str) -> List[str]:
        import re
        
        if not blind_spots_text:
            return []
        
        mentioned = set()
        
        pattern1 = r'([a-zA-Z0-9_/]+\.py::[a-zA-Z0-9_.]+)'
        matches1 = re.findall(pattern1, blind_spots_text)
        mentioned.update(matches1)
        
        pattern2 = r'`([a-zA-Z0-9_/]+\.py)`'
        matches2 = re.findall(pattern2, blind_spots_text)
        mentioned.update(matches2)
        
        pattern3 = r'`([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+){2,})`'
        matches3 = re.findall(pattern3, blind_spots_text)
        for module_path in matches3:
            file_path = module_path.replace('.', '/')
            mentioned.add(file_path)
            mentioned.add(file_path + '.py')
        
        result = list(mentioned)
        self.logger.debug(f"Extracted {len(result)} location fragments from blind_spots: {result[:5]}")
        
        return result


    def _calculate_match_score(self, coverage_loc: str, mentioned_fragments: List[str]) -> int:
        score = 0
        for fragment in mentioned_fragments:
            if fragment in coverage_loc:
                score += len(fragment)
        
        return score



    # unused
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        return None


class SuppletoryLocalizer(BaseLocalizer):

    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""

    module_content_in_block_template = """
### Module: {module_name} ###
```python
{module_content}
```
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        suspicious_contexts, 
        context_level: str = "file"  
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 3000
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.suspicious_contexts = suspicious_contexts
        self.context_level = context_level
        self.suppletory_localize_prompt = suppletory_localize_prompt
        
        self.logger.info(f"SuppletoryLocalizer initialized with context_level={context_level}, {len(suspicious_contexts)} contexts")

    def localize(
        self,
        top_n=1,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
        compress_assign: bool = False,
        total_lines=30,
        prefix_lines=10,
        suffix_lines=10,
    ):
        return self.localize_function_from_compressed_files(
            top_n,
            mock=mock,
            temperature=temperature,
            keep_old_order=keep_old_order,
            compress_assign=compress_assign,
            total_lines=total_lines,
            prefix_lines=prefix_lines,  
            suffix_lines=suffix_lines,
        )

    
    def localize_function_from_compressed_files(
        self,
        top_n=1,
        mock=False,
        temperature=0.0,
        keep_old_order=False,
        compress_assign: bool = False,
        total_lines=30,
        prefix_lines=10,
        suffix_lines=10,
    ):
        from util.api_requests import num_tokens_from_messages
        from util.model import make_model

        if self.context_level == "file":
            file_names = self.suspicious_contexts
            
            from util.test_filter import filter_test_files
            before_filter = len(file_names)
            file_names = filter_test_files(file_names)
            if before_filter != len(file_names):
                self.logger.info(f"Filtered out {before_filter - len(file_names)} test files from suspicious contexts")
            
            file_contents = get_repo_files(self.structure, file_names)

            
            compressed_contents = {
                fn: get_skeleton(
                    code,
                    compress_assign=compress_assign,
                    total_lines=total_lines,
                    prefix_lines=prefix_lines,
                    suffix_lines=suffix_lines,
                )
                for fn, code in file_contents.items()
            }

            contents = [
                self.file_content_in_block_template.format(file_name=fn, file_content=code)
                for fn, code in compressed_contents.items()
            ]
        # 
        else:
            
            
            from util.test_filter import filter_test_locations
            filtered_contexts = filter_test_locations(self.suspicious_contexts)
            if len(filtered_contexts) != len(self.suspicious_contexts):
                self.logger.info(f"Filtered out {len(self.suspicious_contexts) - len(filtered_contexts)} test modules from suspicious contexts")
            
            contents = []
            for module_context in filtered_contexts:
                module_code = self._extract_module_code(module_context)
                if module_code:
                    
                    compressed_code = get_skeleton(
                        module_code,
                        compress_assign=compress_assign,
                        total_lines=total_lines,
                        prefix_lines=prefix_lines,
                        suffix_lines=suffix_lines,
                    )
                    contents.append(
                        self.module_content_in_block_template.format(
                            module_name=module_context,
                            module_content=compressed_code
                        )
                    )
                else:
                    self.logger.warning(f"Could not extract code for module: {module_context}")

        file_contents = "".join(contents)

        message = suppletory_localize_prompt.format(
            problem_statement=self.problem_statement,
            file_contents=file_contents
        ).strip()

        self.logger.info(f"prompting with message:")
        self.logger.info("\n" + message)

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = suppletory_localize_prompt.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        
        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return {}, {"raw_output_loc": ""}, traj


        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=1,
        )

        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)

        # raise ValueError(f"model_found_locs: \n{model_found_locs}")

        
        model_found_locs_separated = [item.strip() for item in model_found_locs[0].split('\n')]

        model_found_locs_separated = self._normalize_init_methods(model_found_locs_separated)
        
        
        from util.test_filter import filter_test_locations
        before_filter = len(model_found_locs_separated)
        model_found_locs_separated = filter_test_locations(model_found_locs_separated)
        if before_filter != len(model_found_locs_separated):
            self.logger.info(f"Filtered out {before_filter - len(model_found_locs_separated)} test locations from model output")

        # model_found_locs_separated = extract_locs_for_files(
        #     model_found_locs, file_names, keep_old_order
        # )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def _normalize_init_methods(self, locations: List[str]) -> List[str]:
        """
        Normalize location list: convert __init__ methods to class level.
        
        This is because modifications to __init__ typically involve:
        1. Parameter descriptions in class docstrings
        2. __init__ method signature
        3. Parent class __init__ calls
        
        These modifications are best annotated at the class level.
        
        Args:
            locations: Original location list
            
        Returns:
            Normalized location list (deduplicated)
        """
        normalized = []
        for loc in locations:
            if '.__init__' in loc:
                normalized_loc = loc.replace('.__init__', '')
                self.logger.debug(f"Normalized {loc} -> {normalized_loc}")
                normalized.append(normalized_loc)
            else:
                normalized.append(loc)
        
        seen = set()
        result = []
        for loc in normalized:
            if loc not in seen:
                seen.add(loc)
                result.append(loc)
        
        return result


    def _extract_module_code(self, module_context: str) -> str:
        if "::" not in module_context:
            self.logger.warning(f"Invalid module context format: {module_context}")
            return ""
        
        file_path, identifier = module_context.split("::", 1)
        
        
        file_content = None
        for fp, content in self.files:
            if fp == file_path:
                file_content = content
                break
        
        if not file_content:
            self.logger.warning(f"File {file_path} not found for module {module_context}")
            return ""
        
        try:
            
            if isinstance(file_content, list):
                lines = file_content
                file_content_str = '\n'.join(file_content)
            else:
                file_content_str = file_content
                lines = file_content.split('\n')
            
            
            tree = ast.parse(file_content_str)
            
            
            target_node = None
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == identifier:
                    target_node = node
                    break
                elif isinstance(node, ast.FunctionDef) and node.name == identifier:
                    target_node = node
                    break
            
            if not target_node:
                self.logger.warning(f"Identifier {identifier} not found in {file_path}")
                return ""
            
            
            start_line = target_node.lineno - 1
            
            if hasattr(target_node, 'end_lineno') and target_node.end_lineno:
                end_line = target_node.end_lineno
            else:
                
                end_line = start_line + 1
                indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
                
                for i in range(start_line + 1, len(lines)):
                    line = lines[i]
                    if line.strip() == "":
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    end_line = i + 1
            
            module_lines = lines[start_line:end_line]
            module_code = '\n'.join(module_lines)
            
            self.logger.debug(f"Extracted {len(module_lines)} lines for module {module_context}")
            return module_code
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error parsing {file_path}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting module code for {module_context}: {e}")
            return ""


class Reranker(BaseLocalizer):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        suspicious_locations,
        context_expansion=False
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 3000
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.suspicious_locations = suspicious_locations
        self.context_expansion = context_expansion

        if self.context_expansion:
            self.rerank_prompt = rerank_prompt_with_context
        else:
            self.rerank_prompt = rerank_prompt
    

    def _extract_code_for_location(self, location: str) -> str:
        if '::' not in location:
            self.logger.warning(f"Invalid location format: {location}")
            return f"[Code not found for {location}]"
        
        file_path, identifier = location.split('::', 1)
        
        
        file_content = None
        for fp, content in self.files:
            if fp == file_path:
                file_content = content
                break
        
        if not file_content:
            self.logger.warning(f"File {file_path} not found")
            return f"[Code not found for {location}]"
        
        try:
            
            if isinstance(file_content, list):
                lines = file_content
                file_content_str = '\n'.join(file_content)
            else:
                file_content_str = file_content
                lines = file_content.split('\n')
            
            
            tree = ast.parse(file_content_str)
            
            
            if '.' in identifier:
                
                class_name, method_name = identifier.split('.', 1)
                return self._extract_method_code(tree, lines, class_name, method_name, location)
            else:
                
                return self._extract_class_or_function_code(tree, lines, identifier, location)
                
        except Exception as e:
            self.logger.error(f"Error extracting code for {location}: {e}")
            return f"[Code extraction failed for {location}]"
    

    def _extract_method_code(self, tree: ast.AST, lines: List[str], class_name: str, method_name: str, location: str) -> str:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                
                for child_node in ast.walk(node):
                    if isinstance(child_node, ast.FunctionDef) and child_node.name == method_name and child_node != node:
                        return self._extract_node_code(child_node, lines)
        
        self.logger.warning(f"Method {class_name}.{method_name} not found")
        return f"[Method {class_name}.{method_name} not found]"
    

    def _extract_class_or_function_code(self, tree: ast.AST, lines: List[str], identifier: str, location: str) -> str:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == identifier:
                
                class_code = self._extract_node_code(node, lines)
                class_lines = class_code.split('\n')
                
                
                if len(class_lines) > 50:
                    self.logger.info(f"Class {identifier} is large ({len(class_lines)} lines), extracting skeleton")
                    return self._extract_class_skeleton(node, lines)
                else:
                    return class_code
        
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == identifier:
                return self._extract_node_code(node, lines)
        
        self.logger.warning(f"Identifier {identifier} not found")
        return f"[Identifier {identifier} not found]"
    

    def _extract_class_skeleton(self, class_node: ast.ClassDef, lines: List[str]) -> str:
        skeleton_lines = []
        
        
        start_line = class_node.lineno - 1
        skeleton_lines.append(lines[start_line])
        
        
        if (class_node.body and 
            isinstance(class_node.body[0], ast.Expr) and 
            isinstance(class_node.body[0].value, (ast.Str, ast.Constant))):
            docstring_node = class_node.body[0]
            doc_start = docstring_node.lineno - 1
            doc_end = getattr(docstring_node, 'end_lineno', doc_start + 1)
            skeleton_lines.extend(lines[doc_start:doc_end])
        
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_start = node.lineno - 1
                
                if node.name == '__init__':
                    
                    method_end = getattr(node, 'end_lineno', method_start + 1)
                    skeleton_lines.extend(lines[method_start:method_end])
                else:
                    
                    skeleton_lines.append(lines[method_start])
                    
                    indent = len(lines[method_start]) - len(lines[method_start].lstrip())
                    skeleton_lines.append(' ' * (indent + 4) + '...')
        
        return '\n'.join(skeleton_lines)
    

    def _extract_node_code(self, node: Union[ast.FunctionDef, ast.ClassDef], lines: List[str]) -> str:
        
        start_line = node.lineno - 1
        
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno
        else:
            
            end_line = start_line + 1
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() == "":
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break
                end_line = i + 1
        
        return '\n'.join(lines[start_line:end_line])


    def localize(self, mock=False):
            from util.api_requests import num_tokens_from_messages
            from util.model import make_model

            locations = list(self.suspicious_locations)
            
            
            from util.test_filter import filter_test_locations
            before_filter = len(locations)
            locations = filter_test_locations(locations)
            if before_filter != len(locations):
                self.logger.info(f"Filtered out {before_filter - len(locations)} test locations from input")
            
            if not locations:
                self.logger.warning("No suspicious locations provided for reranking")
                return [], {}, {}

            self.logger.info(f"Starting reranking for {len(locations)} locations")
            self.logger.info(f"Context expansion: {'enabled' if self.context_expansion else 'disabled'}")
            self.logger.info(f"Locations to rerank: {locations}")

            
            if self.context_expansion:
                
                
                
                location_list_strs = []
                for i, loc in enumerate(locations, 1):
                    location_list_strs.append(f"({i}) {loc}")
                locations_overview = '\n'.join(location_list_strs)
                
                
                locations_context = []
                location_with_context_strs = []
                for i, location in enumerate(locations, 1):
                    code = self._extract_code_for_location(location)
                    locations_context.append(code)
                    self.logger.debug(f"Extracted code for {location}: {len(code)} characters")
                    
                    location_with_context_strs.append(
                        f"### Location {i}: {location} ###\n```python\n{code}\n```"
                    )
                
                locations_details = '\n\n'.join(location_with_context_strs)
                
                
                candidate_locations = f"{locations_overview}\n\n---\n\n{locations_details}"
                
            else:
                
                candidate_locations = '\n'.join(f"- {loc}" for loc in locations)
                

            message = self.rerank_prompt.format(
                problem_statement=self.problem_statement,
                candidate_locations=candidate_locations
            ).strip()

            self.logger.info(f"Prompting with message:\n{message}")
            self.logger.info("=" * 80)

            def message_too_long(message):
                return num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH

            if message_too_long(message):
                self.logger.warning(f"Message too long, applying targeted truncation")
                
                if self.context_expansion:
                    MIN_CODE_LINES = 10  # Minimum retained lines for any location snippet
                    
                    while message_too_long(message) and len(locations) > 0:
                        
                        code_lengths = [len(c.split('\n')) for c in locations_context]
                        max_idx = max(range(len(code_lengths)), key=lambda i: code_lengths[i])
                        max_len = code_lengths[max_idx]
                        
                        if max_len > MIN_CODE_LINES:
                            
                            new_max = max(MIN_CODE_LINES, max_len // 2)
                            code_lines = locations_context[max_idx].split('\n')
                            truncated = '\n'.join(code_lines[:new_max])
                            truncated += f"\n    # ... ({len(code_lines) - new_max} lines omitted) ..."
                            locations_context[max_idx] = truncated
                            self.logger.info(f"Truncated {locations[max_idx]}: {max_len} -> {new_max} lines")
                        else:
                            
                            self.logger.warning(f"Removing location: {locations[max_idx]}")
                            locations.pop(max_idx)
                            locations_context.pop(max_idx)
                        
                        
                        if locations:
                            location_list_strs = [f"({i}) {loc}" for i, loc in enumerate(locations, 1)]
                            locations_overview = '\n'.join(location_list_strs)
                            location_with_context_strs = [
                                f"### Location {i}: {loc} ###\n```python\n{code}\n```"
                                for i, (loc, code) in enumerate(zip(locations, locations_context), 1)
                            ]
                            locations_details = '\n\n'.join(location_with_context_strs)
                            candidate_locations = f"{locations_overview}\n\n---\n\n{locations_details}"
                            message = self.rerank_prompt.format(
                                problem_statement=self.problem_statement,
                                candidate_locations=candidate_locations
                            ).strip()
                else:
                    
                    while len(locations) > 0 and message_too_long(message):
                        locations = locations[:-1]
                        candidate_locations = '\n'.join(f"- {loc}" for loc in locations)
                        message = self.rerank_prompt.format(
                            problem_statement=self.problem_statement,
                            candidate_locations=candidate_locations
                        ).strip()
                
                if not locations:
                    self.logger.error("Cannot fit any locations within context limit")
                    return [], {}, {}
                
                self.logger.info(f"Truncation complete: {len(locations)} locations retained")

            
            if mock:
                self.logger.info("Skipping querying model since mock=True")
                traj = {
                    "prompt": message,
                    "usage": {
                        "prompt_tokens": num_tokens_from_messages(
                            message,
                            self.model_name,
                        ),
                    },
                }
                return {}, {"raw_output_loc": ""}, traj

            model = make_model(
                model=self.model_name,
                backend=self.backend,
                logger=self.logger,
                max_tokens=self.max_tokens,
                batch_size=1,
            )

            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]

            reranked_locations = self.extract_locations(raw_output)
            
            
            from util.test_filter import filter_test_locations
            before_filter = len(reranked_locations)
            reranked_locations = filter_test_locations(reranked_locations)
            if before_filter != len(reranked_locations):
                self.logger.info(f"Filtered out {before_filter - len(reranked_locations)} test locations from model output")

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            self.logger.info(f"==== reranked locs ====")
            for loc in reranked_locations:
                self.logger.info(loc)
            self.logger.info("=" * 80)

            return reranked_locations, {"raw_output_loc": raw_output}, traj

    def extract_locations(self, raw_output: str) -> List[str]:
        if not raw_output:
            self.logger.warning("Empty raw output received for location extraction")
            return []
        
        
        match = re.search(r'```\s*(.*?)\s*```', raw_output, re.DOTALL)
        locations_text = match.group(1).strip() if match else raw_output.strip()
        
        
        lines = [line.strip() for line in locations_text.split('\n') if line.strip()]
        
        locations = []
        for line in lines:
            
            cleaned_line = re.sub(r'^\(?\d+\)?\.?\s*', '', line)
            if cleaned_line:
                locations.append(cleaned_line)
        
        return locations

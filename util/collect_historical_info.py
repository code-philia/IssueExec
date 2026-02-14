#!/usr/bin/env python3
# util/collect_historical_info.py
"""
Collect historical edits related to test functions (optimized version).

Features:
1. Co-modification history between tests and covered code.
2. Commit metadata extraction (message/type).
3. Timeline and frequency statistics.
4. Atomic modification grouping.

Optimizations:
- Batch git operations via `git log -p`.
- Use `git show` to avoid repeated checkout.
- Parallel instance processing.
- Cache AST parsing results.
- Pre-filter to reduce unnecessary analysis.
"""

import os
import sys
import json
import re
import ast
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import subprocess
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from datasets import load_dataset
from nltk.stem import PorterStemmer


class PythonEntityExtractor:
    """Extract entities (class/function/method) from Python sources."""
    
    # Class-level cache for parsed entity mappings.
    _parse_cache: Dict[Tuple[str, str], Dict[int, str]] = {}
    
    @staticmethod
    def parse_source(source: str) -> Dict[int, str]:
        """
        Parse source code and return a mapping: {line_number: entity_name}.
        """
        entities = {}
        
        try:
            # Try Python 3 syntax first.
            try:
                tree = ast.parse(source)
            except SyntaxError:
                # Fallback: try lightweight Python 2 print conversion.
                source_py3 = re.sub(r'\bprint\s+([^(])', r'print(\1)', source)
                try:
                    tree = ast.parse(source_py3)
                except SyntaxError:
                    return entities

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Class definition
                    entity_name = node.name
                    entities[node.lineno] = entity_name
                    
                    # Methods in class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = f"{entity_name}.{item.name}"
                            entities[item.lineno] = method_name
                            
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level function
                    entities[node.lineno] = node.name
                    
        except Exception:
            pass
            
        return entities
    
    @staticmethod
    def parse_file(file_path: str) -> Dict[int, str]:
        """
        Parse a Python file and return {line_number: entity_name}.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            return PythonEntityExtractor.parse_source(source)
        except Exception as e:
            logging.warning(f"Failed to parse {file_path}: {e}")
            return {}
    
    @staticmethod
    def find_entity_at_line(entities: Dict[int, str], line_num: int) -> Optional[str]:
        """Find the closest entity defined at or before `line_num`."""
        # Find max line <= line_num.
        valid_lines = [l for l in entities.keys() if l <= line_num]
        if not valid_lines:
            return None
        closest_line = max(valid_lines)
        return entities[closest_line]

    @staticmethod
    def extract_function_source(source: str, function_name: str) -> Optional[str]:
        """
        Extract function source text by function name.
        """
        try:
            tree = ast.parse(source)
            lines = source.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', None)
                    
                    if end_line:
                        return '\n'.join(lines[start_line:end_line])
                    else:
                        # Python < 3.8 has no `end_lineno`; estimate end by indentation.
                        indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                        for i in range(start_line + 1, len(lines)):
                            line = lines[i]
                            if line.strip() and not line.startswith(' ' * (indent + 1)) and not line.startswith('\t'):
                                if line.strip().startswith('def ') or line.strip().startswith('class '):
                                    return '\n'.join(lines[start_line:i])
                        return '\n'.join(lines[start_line:])
            return None
        except:
            return None
    
    @staticmethod
    def normalize_function_ast(func_source: str) -> Optional[str]:
        """
        Normalize function source into AST dump for semantic comparison.
        """
        try:
            tree = ast.parse(func_source)
            # Ignore position metadata when dumping AST.
            return ast.dump(tree, annotate_fields=False)
        except SyntaxError:
            return None
    
    @staticmethod
    def is_semantic_change(old_source: str, new_source: str, function_name: str) -> bool:
        """
        Return True if change is semantic (not formatting/comment-only).
        """
        old_func = PythonEntityExtractor.extract_function_source(old_source, function_name)
        new_func = PythonEntityExtractor.extract_function_source(new_source, function_name)
        
        if old_func is None and new_func is None:
            # Function unresolved in both sides.
            # Conservative fallback: treat as semantic change.
            logging.debug(f"    Function '{function_name}' not found in both old and new source, assuming semantic change (conservative)")
            return True
        if old_func is None:
            logging.debug(f"    Function '{function_name}' not found in old source (newly added)")
            return True  # function added
        if new_func is None:
            logging.debug(f"    Function '{function_name}' not found in new source (deleted)")
            return True  # function deleted
        
        old_ast = PythonEntityExtractor.normalize_function_ast(old_func)
        new_ast = PythonEntityExtractor.normalize_function_ast(new_func)
        
        if old_ast is None or new_ast is None:
            logging.debug(f"    AST parsing failed for '{function_name}', falling back to text comparison")
            # Fallback to whitespace/comment-insensitive text comparison.
            old_clean = re.sub(r'#.*$', '', old_func, flags=re.MULTILINE)
            old_clean = re.sub(r'\s+', ' ', old_clean).strip()
            new_clean = re.sub(r'#.*$', '', new_func, flags=re.MULTILINE)
            new_clean = re.sub(r'\s+', ' ', new_clean).strip()
            is_different = old_clean != new_clean
            if not is_different:
                logging.debug(f"    Text comparison: no semantic change (whitespace/comment only)")
            return is_different
        
        is_different = old_ast != new_ast
        if not is_different:
            logging.debug(f"    AST comparison: no semantic change (format/comment only)")
        else:
            logging.debug(f"    AST comparison: semantic change detected")
        
        return is_different


class CommitAnalyzer:
    """Analyze git commit metadata and message quality."""
    
    COMMIT_TYPE_PATTERNS = {
        'fix': r'\b(fix|fixed|fixes|bugfix|bug)\b',
        'feat': r'\b(feat|feature|add|added|new)\b',
        'refactor': r'\b(refactor|refactoring|restructure)\b',
        'test': r'\b(test|tests|testing)\b',
        'docs': r'\b(doc|docs|documentation)\b',
        'style': r'\b(style|format|formatting)\b',
        'perf': r'\b(perf|performance|optimize)\b',
        'chore': r'\b(chore|build|ci|release)\b',
    }
    
    # Message patterns to skip (format/style/noise-only commits)
    SKIP_PATTERNS = [
        r'\btypo(s|fix(es)?|graphical\s+error)?\b',  # typo, typos, typofix, typofixes
        r'\bpep\s?8\b',                              # pep8, PEP8, pep 8
        r'\bflake\s?8\b',                            # flake8
        r'\bwhitespace(s)?\b',                       # whitespace, whitespaces
        r'\bindent(s|ed|ing|ation)?\b',              # indent, indentation...
        r'\bcosm[ei]tic(s|al)?\b',                   # cosmetic, cosmit...
        r'\bnit(pick|s)?\b',                         # nit, nitpick
        r'\blint(s|ed|ing|er)?\b',                   # lint, linting, linter
        r'\bspelling\b',                             # spelling
    ]
    
    @staticmethod
    def extract_commit_type(message: str) -> str:
        """Documentation updated for clarity."""
        message_lower = message.lower()
        
        for commit_type, pattern in CommitAnalyzer.COMMIT_TYPE_PATTERNS.items():
            if re.search(pattern, message_lower):
                return commit_type
        
        return 'other'
    
    @staticmethod
    def should_skip_commit(message: str) -> bool:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        message_lower = message.lower()
        
        # Clarified comment.
        if any(re.search(pattern, message_lower) for pattern in CommitAnalyzer.SKIP_PATTERNS):
            return True
        
        # 2) ENcommit message
        if CommitAnalyzer.is_low_information_message(message):
            return True
        
        return False


    @staticmethod
    def is_low_information_message(message: str) -> bool:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        # Clarified comment.
        stemmer = PorterStemmer()

        msg = message.strip().lower()
        if not msg:
            return True

        # Clarified comment.
        msg_clean = re.sub(r'[^a-z0-9\s]', ' ', msg)
        tokens = [t for t in msg_clean.split() if t]

        if not tokens:
            return True

        # Clarified comment.
        stems = [stemmer.stem(token) for token in tokens]

        # Clarified comment.
        generic_single_words = {
            'init', 'initi', 'updat', 'chang', 'refactor', 'cleanup', 'clean', 'fix', 
            'fixes', 'fixing', 'wip', 'temp', 'test', 'typofix', 'typofix', 'minor', 'small'
        }

        generic_words = generic_single_words | {
            'code', 'stuff', 'refactor', 'fixing', 'typofix', 'update', 'test'
        }

        # Clarified comment.
        if len(stems) == 1 and stems[0] in generic_single_words:
            return True

        # Clarified comment.
        if len(stems) <= 3 and all(t in generic_words for t in stems):
            return True

        return False




class GitRepoManager:
    """Documentation updated for clarity."""
    
    def __init__(self, temp_dir: str, instance_id: str = None):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # ENinstanceEN
        self.instance_id = instance_id
        # Clarified comment.
        self._file_content_cache: Dict[Tuple[str, str], str] = {}
        self._entity_cache: Dict[Tuple[str, str], Dict[int, str]] = {}
    
    def clone_repo(self, repo_url: str, commit_hash: str) -> Optional[Path]:
        """Documentation updated for clarity."""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        # Clarified comment.
        if self.instance_id:
            repo_name = f"{repo_name}_{self.instance_id}"
        repo_path = self.temp_dir / repo_name
        
        if repo_path.exists():
            shutil.rmtree(repo_path)
        
        try:
            # Clarified comment.
            subprocess.run(
                ['git', 'clone', f'https://github.com/{repo_url}.git', str(repo_path)],
                check=True,
                capture_output=True,
                timeout=300
            )
            
            # ENcommit
            subprocess.run(
                ['git', 'checkout', commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            
            return repo_path
            
        except Exception as e:
            logging.error(f"Failed to clone repo {repo_url}: {e}")
            return None
    
    def get_commit_history(self, repo_path: Path, file_path: str, 
                        since_commit: Optional[str] = None,
                        filter_non_semantic: bool = True) -> List[Dict]:
        """
        # Documentation updated for clarity.
        
        Args:
            # Documentation updated for clarity.
            # Documentation updated for clarity.
            # Documentation updated for clarity.
            # Documentation updated for clarity.
        """
        try:
            cmd = ['git', 'log', '--follow', '--format=%H|%at|%s', '--', file_path]
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            skipped_count = 0
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('|', 2)
                if len(parts) == 3:
                    commit_hash, timestamp, message = parts
                    
                    # Clarified comment.
                    if filter_non_semantic and CommitAnalyzer.should_skip_commit(message):
                        skipped_count += 1
                        logging.debug(f"Skipping commit {commit_hash[:8]} by message filter: {message[:60]}")
                        continue
                    
                    commits.append({
                        'commit_hash': commit_hash,
                        'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat() + 'Z',
                        'commit_message': message,
                        'commit_type': CommitAnalyzer.extract_commit_type(message)
                    })
            
            if skipped_count > 0:
                logging.debug(f"Filtered {skipped_count} non-semantic commits for {file_path}")
            
            if not commits:
                logging.warning(f"No commit history found for {file_path}")
            
            return commits
            
        except subprocess.CalledProcessError as e:
            logging.warning(f"Git log failed for {file_path}: {e.stderr}")
            return []
        except Exception as e:
            logging.warning(f"Failed to get commit history for {file_path}: {e}")
            return []


    def get_commit_history_with_diffs(self, repo_path: Path, file_path: str,
                                      filter_non_semantic: bool = True) -> Tuple[List[Dict], Dict[str, Dict[str, List[Tuple[int, int]]]], Dict[str, List[str]]]:
        """
        # Documentation updated for clarity.
        
        Returns:
            (commits, all_diffs, all_files)
            # Documentation updated for clarity.
            - all_diffs: {commit_hash: {file_path: [(start, end), ...]}}
            - all_files: {commit_hash: [file_path, ...]}
        """
        try:
            # Clarified comment.
            # Clarified comment.
            cmd = [
                'git', 'log', '--follow',
                '--format=COMMIT_START%n%H|%at|%s',
                '-p', '--name-only',
                '--', file_path
            ]
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logging.warning(f"git log -p failed for {file_path}: {result.stderr}")
                return [], {}, {}
            
            commits = []
            all_diffs = {}
            all_files = {}
            skipped_count = 0
            
            # Clarified comment.
            commit_blocks = result.stdout.split('COMMIT_START\n')
            
            for block in commit_blocks:
                if not block.strip():
                    continue
                
                lines = block.split('\n')
                if not lines:
                    continue
                
                # Clarified comment.
                header_line = lines[0]
                parts = header_line.split('|', 2)
                if len(parts) != 3:
                    continue
                
                commit_hash, timestamp, message = parts
                
                # Clarified comment.
                if filter_non_semantic and CommitAnalyzer.should_skip_commit(message):
                    skipped_count += 1
                    logging.debug(f"Skipping commit {commit_hash[:8]} by message filter: {message[:60]}")
                    continue
                
                commits.append({
                    'commit_hash': commit_hash,
                    'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat() + 'Z',
                    'commit_message': message,
                    'commit_type': CommitAnalyzer.extract_commit_type(message)
                })
                
                # Clarified comment.
                diff_content = '\n'.join(lines[1:])
                all_diffs[commit_hash] = self._parse_diff_output(diff_content)
                
                # Clarified comment.
                # Clarified comment.
                all_files[commit_hash] = list(all_diffs[commit_hash].keys())
                
                # Clarified comment.
                in_diff = False
                for line in lines[1:]:
                    if line.startswith('diff --git'):
                        in_diff = True
                    elif line.startswith('COMMIT_START') or (not in_diff and line and not line.startswith((' ', '+', '-', '@', '\\', 'index', 'new', 'deleted', 'old', 'similarity', 'rename', 'Binary'))):
                        # Clarified comment.
                        if line.endswith('.py') and line not in all_files[commit_hash]:
                            all_files[commit_hash].append(line)
            
            if skipped_count > 0:
                logging.debug(f"Filtered {skipped_count} non-semantic commits for {file_path}")
            
            return commits, all_diffs, all_files
            
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout while getting commit history for {file_path}")
            return [], {}, {}
        except Exception as e:
            logging.warning(f"Failed to get commit history with diffs for {file_path}: {e}")
            return [], {}, {}


    # Clarified comment.
    def get_batch_commit_diffs(self, repo_path: Path, commit_hashes: List[str]) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        if not commit_hashes:
            return {}
        
        result = {}
        
        # Clarified comment.
        batch_size = 50
        for i in range(0, len(commit_hashes), batch_size):
            batch = commit_hashes[i:i + batch_size]
            
            for commit_hash in batch:
                try:
                    diff_result = subprocess.run(
                        ['git', 'diff', f'{commit_hash}^', commit_hash, '--unified=0'],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if diff_result.returncode != 0:
                        continue
                    
                    result[commit_hash] = self._parse_diff_output(diff_result.stdout)
                    
                except Exception as e:
                    logging.debug(f"Failed to get diff for commit {commit_hash}: {e}")
                    result[commit_hash] = {}
        
        return result
    
    def _parse_diff_output(self, diff_output: str) -> Dict[str, List[Tuple[int, int]]]:
        """Documentation updated for clarity."""
        modified_files = {}
        current_file = None
        
        for line in diff_output.split('\n'):
            # Clarified comment.
            if line.startswith('+++'):
                file_match = re.match(r'\+\+\+ b/(.+)', line)
                if file_match:
                    current_file = file_match.group(1)
                    if current_file not in modified_files:
                        modified_files[current_file] = []
            
            # Clarified comment.
            elif line.startswith('@@') and current_file:
                match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    modified_files[current_file].append((start, start + count - 1))
        
        return modified_files
    
    def get_commit_diff(self, repo_path: Path, commit_hash: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        result = self.get_batch_commit_diffs(repo_path, [commit_hash])
        return result.get(commit_hash, {})
    
    def get_files_in_commit(self, repo_path: Path, commit_hash: str) -> List[str]:
        """Documentation updated for clarity."""
        try:
            result = subprocess.run(
                ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return [f for f in result.stdout.strip().split('\n') if f]
            
        except Exception as e:
            logging.warning(f"Failed to get files in commit {commit_hash}: {e}")
            return []
    
    # Clarified comment.
    def get_batch_files_in_commits(self, repo_path: Path, commit_hashes: List[str]) -> Dict[str, List[str]]:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        result = {}
        
        for commit_hash in commit_hashes:
            try:
                cmd_result = subprocess.run(
                    ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if cmd_result.returncode == 0:
                    files = [f for f in cmd_result.stdout.strip().split('\n') if f]
                    result[commit_hash] = files
                else:
                    result[commit_hash] = []
                    
            except Exception:
                result[commit_hash] = []
        
        return result
    
    # Clarified comment.
    def get_file_content_at_commit(self, repo_path: Path, commit_hash: str, file_path: str) -> Optional[str]:
        """
        # Documentation updated for clarity.
        """
        cache_key = (commit_hash, file_path)
        
        # Clarified comment.
        if cache_key in self._file_content_cache:
            return self._file_content_cache[cache_key]
        
        try:
            result = subprocess.run(
                ['git', 'show', f'{commit_hash}:{file_path}'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                content = result.stdout
                # Clarified comment.
                if len(self._file_content_cache) < 1000:
                    self._file_content_cache[cache_key] = content
                return content
            return None
            
        except Exception as e:
            logging.debug(f"Failed to get file content for {file_path} at {commit_hash}: {e}")
            return None
    
    # Clarified comment.
    def get_file_content_before_commit(self, repo_path: Path, commit_hash: str, file_path: str) -> Optional[str]:
        """
        # Documentation updated for clarity.
        """
        return self.get_file_content_at_commit(repo_path, f'{commit_hash}^', file_path)
    
    # Clarified comment.
    def get_entities_at_commit(self, repo_path: Path, commit_hash: str, file_path: str) -> Dict[int, str]:
        """
        # Documentation updated for clarity.
        """
        cache_key = (commit_hash, file_path)
        
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        content = self.get_file_content_at_commit(repo_path, commit_hash, file_path)
        if content is None:
            return {}
        
        entities = PythonEntityExtractor.parse_source(content)
        
        # Clarified comment.
        if len(self._entity_cache) < 2000:
            self._entity_cache[cache_key] = entities
        
        return entities
    
    def find_function_init_commit(self, repo_path: Path, file_path: str, function_name: str) -> Optional[Dict]:
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        try:
            search_pattern = f"def {function_name}("
            
            logging.debug(f"Searching init commit for function '{function_name}' in {file_path}")
            
            # Clarified comment.
            # Clarified comment.
            cmd = [
                'git', 'log', '-S', search_pattern,
                '--format=COMMIT_MARKER|%H|%at|%s',
                '-p', '--reverse',
                '--', file_path
            ]
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode != 0:
                logging.warning(f"git log -p -S failed for {function_name}")
                return None
            
            # Clarified comment.
            output = result.stdout
            commit_blocks = output.split('COMMIT_MARKER|')
            
            logging.debug(f"Found {len(commit_blocks) - 1} candidate commits")
            
            # Clarified comment.
            move_patterns = [
                r'\bmoves?\s+(project|directory|folder|files?)\b',
                r'\bmoves?\s+\S+\s+(to|from)\s+\S+',
                r'\bmoves?\s+\S+\s+(out\s+of|into)\s+\S+',
                r'\brename\s+(project|directory|folder)\b',
                r'\brelocate\s+(project|directory|folder)\b',
                r'\bmigrate\s+(project|directory|folder)\b',
            ]
            
            for block in commit_blocks:
                if not block.strip():
                    continue
                
                lines = block.split('\n')
                header = lines[0]
                parts = header.split('|', 2)
                if len(parts) != 3:
                    continue
                
                commit_hash, timestamp, message = parts
                message_lower = message.lower()
                diff_content = '\n'.join(lines[1:])
                
                # Clarified comment.
                if CommitAnalyzer.should_skip_commit(message):
                    logging.debug(f"Skipping commit {commit_hash[:8]}: non-semantic type - '{message[:60]}'")
                    continue
                
                # Clarified comment.
                is_move_commit = any(re.search(pattern, message_lower) for pattern in move_patterns)
                
                # Clarified comment.
                has_added_func_def = False
                has_removed_func_def = False
                
                for diff_line in diff_content.split('\n'):
                    # Clarified comment.
                    if diff_line.startswith('+++') or diff_line.startswith('---'):
                        continue
                    if diff_line.startswith('+') and search_pattern in diff_line:
                        has_added_func_def = True
                    if diff_line.startswith('-') and search_pattern in diff_line:
                        has_removed_func_def = True
                
                # Clarified comment.
                if is_move_commit and not has_added_func_def:
                    logging.debug(f"Skipping commit {commit_hash[:8]}: file move without adding function")
                    continue
                
                # Clarified comment.
                if has_added_func_def and has_removed_func_def:
                    logging.debug(f"Skipping commit {commit_hash[:8]}: function was moved/renamed within file")
                    continue
                
                # Clarified comment.
                if not has_added_func_def:
                    logging.debug(f"Skipping commit {commit_hash[:8]}: no '+def {function_name}(' in diff")
                    continue
                
                # Clarified comment.
                is_new_file = 'new file mode' in diff_content
                is_rename = 'rename from' in diff_content or 'similarity index' in diff_content
                
                if is_rename and not is_new_file:
                    # Clarified comment.
                    # Clarified comment.
                    # Clarified comment.
                    logging.debug(f"Commit {commit_hash[:8]} is rename but function was added in this commit")
                
                # Clarified comment.
                # ENdiffEN
                file_count = diff_content.count('diff --git')
                if file_count > 50:
                    logging.debug(f"Skipping commit {commit_hash[:8]}: too many files changed ({file_count})")
                    continue
                
                # Clarified comment.
                init_commit = {
                    'commit_hash': commit_hash,
                    'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat() + 'Z',
                    'commit_message': message,
                    'commit_type': CommitAnalyzer.extract_commit_type(message)
                }
                logging.info(f"Found init commit for '{function_name}': {commit_hash[:8]} - {message[:60]}")
                return init_commit
            
            logging.warning(f"No verified init commit found for function '{function_name}' in {file_path}")
            return None
            
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout while searching init commit for {function_name}")
            return None
        except Exception as e:
            logging.warning(f"Failed to find init commit for {function_name}: {e}")
            return None


    def is_semantic_modification(self, repo_path: Path, commit_hash: str, 
                                    file_path: str, function_name: str) -> bool:
        """
        # Documentation updated for clarity.
        """
        try:
            logging.debug(f"Checking semantic modification for '{function_name}' in commit {commit_hash[:8]}")
            
            old_content = self.get_file_content_at_commit(repo_path, f'{commit_hash}^', file_path)
            new_content = self.get_file_content_at_commit(repo_path, commit_hash, file_path)
            
            if old_content is None and new_content is None:
                logging.debug(f"  Both old and new content are None, not a semantic change")
                return False
            if old_content is None:
                logging.debug(f"  Old content is None (file created), is semantic change")
                return True
            if new_content is None:
                logging.debug(f"  New content is None (file deleted), is semantic change")
                return True
            
            is_semantic = PythonEntityExtractor.is_semantic_change(old_content, new_content, function_name)
            
            if is_semantic:
                logging.debug(f"  Commit {commit_hash[:8]} IS a semantic modification to '{function_name}'")
            else:
                logging.debug(f"  Commit {commit_hash[:8]} is NOT a semantic modification to '{function_name}' (format/comment only)")
            
            return is_semantic
            
        except Exception as e:
            logging.debug(f"Failed to check semantic modification for {function_name} in {commit_hash[:8]}: {e}")
            return True  # Clarified comment.
    
    def clear_cache(self):
        """Documentation updated for clarity."""
        self._file_content_cache.clear()
        self._entity_cache.clear()


class HistoricalInfoCollector:
    """Documentation updated for clarity."""
    
    def __init__(self, output_dir: str, log_dir: str, instance_id: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ENinstance_idEN
        # ENIO
        self.git_manager = GitRepoManager('/tmp/swe_bench_repos', instance_id)
        self.entity_extractor = PythonEntityExtractor()

    
    def setup_logger(self, instance_id: str) -> logging.Logger:
        """Documentation updated for clarity."""
        logger = logging.getLogger(instance_id)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False  # Clarified comment.
        
        # ENhandler
        fh = logging.FileHandler(self.log_dir / f'{instance_id}.log', mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Clarified comment.
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        # Clarified comment.
        
        return logger
    
    def _prefetch_file_contents(self, repo_path: Path, file_path: str, commit_hashes: List[str]):
        """
        # Documentation updated for clarity.
        # Documentation updated for clarity.
        """
        if not commit_hashes:
            return
        
        # Clarified comment.
        refs_to_fetch = []
        for commit_hash in commit_hashes:
            refs_to_fetch.append(f"{commit_hash}:{file_path}")
            refs_to_fetch.append(f"{commit_hash}^:{file_path}")
        
        # Clarified comment.
        try:
            # Clarified comment.
            input_data = '\n'.join(refs_to_fetch)
            
            result = subprocess.run(
                ['git', 'cat-file', '--batch'],
                cwd=repo_path,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logging.debug(f"git cat-file --batch failed, falling back to individual fetches")
                return
            
            # Clarified comment.
            # Clarified comment.
            output = result.stdout
            lines = output.split('\n')
            
            i = 0
            ref_idx = 0
            while i < len(lines) and ref_idx < len(refs_to_fetch):
                header = lines[i]
                ref = refs_to_fetch[ref_idx]
                
                # Clarified comment.
                if ':' in ref:
                    commit_ref, fpath = ref.rsplit(':', 1)
                    # Clarified comment.
                    cache_key = (commit_ref, fpath)
                else:
                    ref_idx += 1
                    i += 1
                    continue
                
                if 'missing' in header:
                    # Clarified comment.
                    i += 1
                    ref_idx += 1
                    continue
                
                # Clarified comment.
                parts = header.split()
                if len(parts) >= 3 and parts[1] == 'blob':
                    try:
                        size = int(parts[2])
                        # Clarified comment.
                        content_lines = []
                        remaining = size
                        i += 1
                        while remaining > 0 and i < len(lines):
                            line = lines[i]
                            content_lines.append(line)
                            remaining -= len(line) + 1  # +1 for newline
                            i += 1
                        
                        content = '\n'.join(content_lines)
                        # Clarified comment.
                        if len(self.git_manager._file_content_cache) < 1000:
                            self.git_manager._file_content_cache[cache_key] = content
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
                
                ref_idx += 1
                
        except subprocess.TimeoutExpired:
            logging.debug("Timeout during batch prefetch, will use individual fetches")
        except Exception as e:
            logging.debug(f"Batch prefetch failed: {e}, will use individual fetches")


    # Clarified comment.
    def extract_entities_from_diff_optimized(self, repo_path: Path, commit_hash: str, 
                                            file_path: str, modified_lines: List[Tuple[int, int]]) -> Set[str]:
        """
        # Documentation updated for clarity.
        """
        entities = set()
        
        if not file_path.endswith('.py'):
            return entities
        
        try:
            # Clarified comment.
            entity_map = self.git_manager.get_entities_at_commit(
                repo_path, f'{commit_hash}^', file_path
            )
            
            if not entity_map:
                return entities
            
            # Clarified comment.
            for start, end in modified_lines:
                for line_num in range(start, end + 1):
                    entity = self.entity_extractor.find_entity_at_line(entity_map, line_num)
                    if entity:
                        entities.add(f"{file_path}::{entity}")
            
        except Exception as e:
            logging.debug(f"Failed to extract entities from {file_path}: {e}")
        
        return entities

    # Clarified comment.
    def extract_entities_from_diff(self, repo_path: Path, commit_hash: str, 
                                file_path: str, modified_lines: List[Tuple[int, int]]) -> Set[str]:
        """Documentation updated for clarity."""
        return self.extract_entities_from_diff_optimized(repo_path, commit_hash, file_path, modified_lines)

    def collect_for_test(self, logger: logging.Logger, repo_path: Path, 
                        test_function: str, covered_entities: List[str],
                        base_commit: str) -> Dict:
        """Documentation updated for clarity."""
        logger.info(f"Collecting history for test: {test_function}")
        
        result = {
            'test_function': test_function,
            'covered_entities': covered_entities,
            'init_commit': None,  # Clarified comment.
            'co_modifications': [],
            'test_modification_history': [],
            'co_occurrence_timeline': {},
            'modification_groups': [],
            'statistics': {
                'total_test_modifications': 0,
                'total_co_modifications': 0,
                'co_modified_entities_count': 0,
                'avg_modification_group_size': 0.0,
                'core_entities_count': 0,
                'extended_entities_count': 0
            }
        }
        
        # Clarified comment.
        test_file = test_function.split('::')[0]

        # Clarified comment.
        test_file_content = self.git_manager.get_file_content_at_commit(repo_path, base_commit, test_file)
        if test_file_content is None:
            logger.warning(f"Test file does not exist at base_commit: {test_file}")
            return result
        
        logger.debug(f"Test file exists: {test_file}")

        # ENAEN：ENcommitEN、diffEN
        test_commits, all_diffs, all_files = self.git_manager.get_commit_history_with_diffs(
            repo_path, test_file
        )
        logger.info(f"Found {len(test_commits)} commits for test file (with diffs pre-fetched)")
        
        # Clarified comment.
        test_func_name = test_function.split('::')[-1] if '::' in test_function else None
        
        # ENinit commit
        if test_func_name:
            init_commit = self.git_manager.find_function_init_commit(repo_path, test_file, test_func_name)
            result['init_commit'] = init_commit
            if init_commit:
                logger.info(f"Found init commit for {test_func_name}: {init_commit['commit_hash'][:8]}")
            else:
                logger.warning(f"Could not find init commit for {test_func_name}")
        
        # Clarified comment.
        if test_func_name:
            logger.info(f"Starting semantic filtering for {len(test_commits)} commits...")
            semantic_commits = []
            filtered_commits = []
            
            # Clarified comment.
            commits_to_check = [c['commit_hash'] for c in test_commits]
            self._prefetch_file_contents(repo_path, test_file, commits_to_check)
            
            for commit_info in test_commits:
                commit_hash = commit_info['commit_hash']
                commit_msg = commit_info['commit_message']
                
                is_semantic = self.git_manager.is_semantic_modification(repo_path, commit_hash, test_file, test_func_name)
                
                if is_semantic:
                    semantic_commits.append(commit_info)
                    logger.debug(f"KEPT semantic commit: {commit_hash[:8]} - {commit_msg[:60]}")
                else:
                    filtered_commits.append(commit_info)
                    logger.info(f"FILTERED non-semantic commit: {commit_hash[:8]} - {commit_msg[:60]}")
            
            # Clarified comment.
            logger.info(f"Semantic filtering complete:")
            logger.info(f"  - Total commits analyzed: {len(test_commits)}")
            logger.info(f"  - Semantic commits (kept): {len(semantic_commits)}")
            logger.info(f"  - Non-semantic commits (filtered): {len(filtered_commits)}")
            
            if filtered_commits:
                logger.info(f"  - Filtered commits list:")
                for fc in filtered_commits:
                    logger.info(f"      {fc['commit_hash'][:8]} | {fc['timestamp'][:10]} | {fc['commit_message'][:50]}")
            
            test_commits = semantic_commits
        
        result['test_modification_history'] = test_commits
        result['statistics']['total_test_modifications'] = len(test_commits)
        
        if not test_commits:
            return result
        
        # ENcovered entitiesEN
        covered_set = set(covered_entities)
        
        # Clarified comment.
        covered_files = set()
        for entity in covered_entities:
            if '::' in entity:
                covered_files.add(entity.split('::')[0])
        
        # Clarified comment.
        logger.debug(f"Using pre-fetched diff info for {len(test_commits)} commits")
        
        first_test_commit_time = test_commits[-1]['timestamp'] if test_commits else None
        co_modified_entities = set()
        
        for commit_info in test_commits:
            commit_hash = commit_info['commit_hash']
            logger.debug(f"Analyzing commit: {commit_hash}")
            
            # Clarified comment.
            modified_files_in_commit = all_files.get(commit_hash, [])
            file_diffs = all_diffs.get(commit_hash, {})
            
            # Clarified comment.
            modified_py_files = [f for f in modified_files_in_commit if f.endswith('.py')]
            
            # ENcovered filesEN
            has_covered_file = any(f in covered_files for f in modified_py_files)
            has_test_file = test_file in modified_files_in_commit
            
            if not has_test_file and not has_covered_file:
                # Clarified comment.
                continue
            
            # ENcommitEN
            modified_entities_in_commit = set()
            
            for file_path, line_ranges in file_diffs.items():
                if not file_path.endswith('.py'):
                    continue
                
                # Clarified comment.
                if file_path != test_file and file_path not in covered_files:
                    continue
                
                entities = self.extract_entities_from_diff_optimized(
                    repo_path, commit_hash, file_path, line_ranges
                )
                modified_entities_in_commit.update(entities)
            
            # ENcovered_entitiesEN
            covered_modified = modified_entities_in_commit & covered_set
            
            if test_function in modified_entities_in_commit or has_test_file:
                # ENcommit
                if covered_modified:
                    # Clarified comment.
                    co_mod_record = {
                        'commit_hash': commit_hash,
                        'timestamp': commit_info['timestamp'],
                        'modified_entities': sorted(list(covered_modified | {test_function})),
                        'commit_message': commit_info['commit_message'],
                        'commit_type': commit_info['commit_type']
                    }
                    result['co_modifications'].append(co_mod_record)
                    
                    # Clarified comment.
                    result['modification_groups'].append({
                        'commit_hash': commit_hash,
                        'timestamp': commit_info['timestamp'],
                        'commit_message': commit_info['commit_message'],
                        'commit_type': commit_info['commit_type'],
                        'entities_modified_together': sorted(list(covered_modified | {test_function})),
                        'group_size': len(covered_modified) + 1
                    })
                    
                    # Clarified comment.
                    for entity in covered_modified:
                        if entity not in result['co_occurrence_timeline']:
                            result['co_occurrence_timeline'][entity] = {
                                'first_co_modification': commit_info['timestamp'],
                                'is_initial_coverage': (commit_info['timestamp'] == first_test_commit_time),
                                'co_modification_count': 0
                            }
                        result['co_occurrence_timeline'][entity]['co_modification_count'] += 1
                    
                    co_modified_entities.update(covered_modified)
        
        # Clarified comment.
        result['statistics']['total_co_modifications'] = len(result['co_modifications'])
        result['statistics']['co_modified_entities_count'] = len(co_modified_entities)
        
        if result['modification_groups']:
            avg_size = sum(g['group_size'] for g in result['modification_groups']) / len(result['modification_groups'])
            result['statistics']['avg_modification_group_size'] = round(avg_size, 2)
        
        result['statistics']['core_entities_count'] = sum(
            1 for info in result['co_occurrence_timeline'].values() 
            if info['is_initial_coverage']
        )
        result['statistics']['extended_entities_count'] = sum(
            1 for info in result['co_occurrence_timeline'].values() 
            if not info['is_initial_coverage']
        )
        
        logger.info(f"Completed collection for {test_function}")
        logger.info(f"Statistics: {result['statistics']}")
        
        return result


    def process_instance(self, instance: Dict, coverage_graph: Dict):
        """Documentation updated for clarity."""
        instance_id = instance['instance_id']
        logger = self.setup_logger(instance_id)
        
        logger.info(f"{'='*80}")
        logger.info(f"Processing instance: {instance_id}")
        logger.info(f"{'='*80}")
        
        try:
            # Clarified comment.
            repo = instance['repo']
            base_commit = instance['base_commit']
            
            logger.info(f"Cloning repository: {repo}")
            logger.info(f"Base commit: {base_commit}")
            
            repo_path = self.git_manager.clone_repo(repo, base_commit)
            if not repo_path:
                logger.error("Failed to clone repository")
                return None
            
            logger.info(f"Repository cloned to: {repo_path}")
            
            # Clarified comment.
            results = {}
            
            for test_function, test_data in coverage_graph.items():
                covered_entities = test_data['nodes']
                
                test_result = self.collect_for_test(
                    logger, repo_path, test_function, covered_entities, base_commit
                )
                
                results[test_function] = test_result
            
            # Clarified comment.
            self.git_manager.clear_cache()
            
            # Clarified comment.
            output_file = self.output_dir / f'{instance_id}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"Successfully processed {len(results)} test functions")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
            return None
        
        finally:
            # Clarified comment.
            logger.info("Cleaning up...")
            try:
                if repo_path and repo_path.exists():
                    shutil.rmtree(repo_path)
                    logger.info(f"Removed cloned repo: {repo_path}")
            except NameError:
                # Clarified comment.
                pass
            except Exception as e:
                logger.warning(f"Failed to remove repo: {e}")


# Clarified comment.
def process_instance_parallel(args: Tuple[Dict, str, str, str]) -> Optional[str]:
    """
    # Documentation updated for clarity.
    args: (instance, coverage_file_path, output_dir, log_dir)
    """
    instance, coverage_file_path, output_dir, log_dir = args

    # Clarified comment.
    try:
        with open(coverage_file_path, 'r', encoding='utf-8') as f:
            coverage_graph = json.load(f)
    except Exception as e:
        # Clarified comment.
        instance_id = instance.get('instance_id', 'UNKNOWN')
        print(f"[WARN] Failed to load coverage graph for {instance_id}: {coverage_file_path} ({e})", flush=True)
        return None

    # ENcollector，ENinstance_idEN
    instance_id = instance['instance_id']
    collector = HistoricalInfoCollector(output_dir, log_dir, instance_id)

    return collector.process_instance(instance, coverage_graph)



def main(swe_bench_path: str, coverage_graph_path: str, output_dir: str = 'historical_information',
         num_workers: int = None):
    """Documentation updated for clarity."""
    print("="*80)
    print("Historical Information Collection Tool (Optimized)")
    print("="*80)
    
    # Clarified comment.
    if num_workers is None:
        # Clarified comment.
        cpu_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(cpu_workers, 16)
    
    print(f"Using {num_workers} parallel workers")
    
    # Clarified comment.
    log_dir = 'logs'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ENSWE-benchEN
    print(f"\nLoading SWE-bench data from: {swe_bench_path}")
    swe_bench_data = load_dataset(swe_bench_path)
    
    # Clarified comment.
    test_data = swe_bench_data['test']
    print(f"Loaded {len(test_data)} instances")
    
    # ENcoverage graphs
    coverage_graph_dir = Path(coverage_graph_path)
    
    # Clarified comment.
    tasks = []
    total_instances = len(test_data)

    for idx, instance in enumerate(test_data, 1):
        instance_id = instance['instance_id']
        coverage_file = coverage_graph_dir / f'{instance_id}.json'

        # Clarified comment.
        if idx % 10 == 0 or idx == total_instances:
            print(f"Preparing tasks: {idx}/{total_instances}", flush=True)

        if not coverage_file.exists():
            print(f"Skipping {instance_id}: coverage graph not found")
            continue

        # ENinstanceEN
        instance_dict = dict(instance)

        # Clarified comment.
        tasks.append((instance_dict, str(coverage_file), output_dir, log_dir))

    
    print(f"\nPrepared {len(tasks)} tasks for processing")
    
    # Clarified comment.
    processed = 0
    failed = 0
    
    if num_workers > 1:
        print(f"\nStarting parallel processing with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_instance_parallel, task): task[0]['instance_id'] 
                      for task in tasks}
            
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed += 1
                        print(f"[{processed + failed}/{len(tasks)}] Completed: {instance_id}")
                    else:
                        failed += 1
                        print(f"[{processed + failed}/{len(tasks)}] Failed: {instance_id}")
                except Exception as e:
                    failed += 1
                    print(f"[{processed + failed}/{len(tasks)}] Error processing {instance_id}: {e}")
    else:
        # Clarified comment.
        print("\nStarting sequential processing...")
        collector = HistoricalInfoCollector(output_dir, log_dir)

        for task in tasks:
            instance, coverage_file_path, _, _ = task
            instance_id = instance['instance_id']
            print(f"\n[{processed + failed + 1}/{len(tasks)}] Processing: {instance_id}", flush=True)

            try:
                with open(coverage_file_path, 'r', encoding='utf-8') as f:
                    coverage_graph = json.load(f)
            except Exception as e:
                failed += 1
                print(f"[{processed + failed}/{len(tasks)}] Failed to load coverage for {instance_id}: {e}", flush=True)
                continue

            result = collector.process_instance(instance, coverage_graph)
            if result:
                processed += 1
            else:
                failed += 1

    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Processed: {processed} instances")
    print(f"Failed: {failed} instances")
    print(f"Results saved to: {output_dir}/")
    print(f"Logs saved to: {log_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python collect_historical_info_optimized.py <swe_bench_path> <coverage_graph_path> [output_dir] [num_workers]")
        print("  num_workers: number of parallel workers (default: CPU count - 1)")
        sys.exit(1)
    
    swe_bench_path = sys.argv[1]
    coverage_graph_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'historical_information'
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    main(swe_bench_path, coverage_graph_path, output_dir, num_workers)

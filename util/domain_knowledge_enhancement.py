#!/usr/bin/env python3
# util/domain_knowledge_enhancement.py
"""
Domain Knowledge Enhancement Script

Extract domain-knowledge tokens from historical information (init_commit and co_modification)
for each test function, following the designed pipeline:
    Step 0: Pre-filtering (low-quality commit, frequency threshold)
    Step 1: Code entity extraction (naming pattern recognition)
    Step 2: Normalization + Deduplication (against test_func tokens)
"""

import argparse
import json
import logging
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Set, Optional

# ==================== Stemming Utilities ====================
try:
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import words as nltk_words
    STEMMER = PorterStemmer()

    # Try loading the "words" corpus; download automatically if missing
    try:
        ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
    except LookupError:
        nltk.download('words', quiet=True)
        ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

    # Precompute stemmed dictionary for matching inflections/variants
    ENGLISH_WORD_STEMS = set(STEMMER.stem(w) for w in ENGLISH_WORDS)

    # Remove prints; use module-level flags only
    _NLTK_LOADED = True
except ImportError:
    STEMMER = None
    ENGLISH_WORDS = set()
    ENGLISH_WORD_STEMS = set()
    _NLTK_LOADED = False


# ==================== Configuration Constants ====================

# Step 0: Low-quality init_commit prefixes
LOW_QUALITY_PREFIXES = {
    'MAINT', 'COSMIT', 'DOC', 'TST', 'TEST', 'MISC',
    'CHORE', 'STYLE', 'CI', 'BUILD', 'CLEANUP', 'TYPO'
}

# Step 0: Low-quality init_commit patterns
LOW_QUALITY_PATTERNS = [
    r'^(more\s+tests?|basic\s+testing|added?\s+tests?|tests?\s+added)$',
    r'^(fix|bug|test|wip|todo)$',
    r'^(minor|small|tiny)(\s+(fix|change|update|tweak))?$',
    r'^\s*$',
]

# Step 1: Code-domain stopwords
CODE_STOPWORDS = {
    # Testing related
    'test', 'tests', 'testing', 'check', 'checks', 'checking', 'assert', 'verify',
    # Common utilities
    'utils', 'util', 'utility', 'utilities', 'helper', 'helpers', 'common', 'shared',
    # Basic setup
    'base', 'basic', 'init', 'initialize', 'setup',
    # Python keywords / common params
    'self', 'cls', 'args', 'kwargs', 'super', 'lambda',
    # Project / language identifiers
    'sklearn', 'scikit', 'learn', 'py', 'python', 'numpy', 'scipy',
    # Generic verbs
    'get', 'set', 'make', 'create', 'build', 'run', 'execute', 'call', 'do',
    'add', 'remove', 'delete', 'update', 'insert', 'append',
    'load', 'save', 'read', 'write', 'parse', 'format',
    # Generic nouns
    'file', 'files', 'path', 'paths', 'dir', 'directory', 'folder',
    'input', 'output', 'result', 'results', 'response', 'request',
    'data', 'value', 'values', 'item', 'items', 'element', 'elements',
    'name', 'names', 'key', 'keys', 'index', 'indices',
    'type', 'types', 'kind', 'mode', 'option', 'options',
    'size', 'length', 'count', 'number', 'num', 'total',
    # Errors / logging
    'error', 'errors', 'exception', 'exceptions', 'warning', 'warnings',
    'info', 'debug', 'log', 'logger', 'logging',
    # Code structure
    'func', 'function', 'functions', 'method', 'methods',
    'class', 'classes', 'module', 'modules', 'package', 'packages',
    'param', 'params', 'parameter', 'parameters',
    'arg', 'argument', 'arguments',
    'return', 'returns', 'yield', 'yields',
    'private', 'public', 'protected', 'internal', 'external',
    # Booleans / nulls
    'true', 'false', 'none', 'null', 'empty',
    # Basic types
    'str', 'string', 'int', 'integer', 'float', 'double', 'bool', 'boolean',
    'list', 'dict', 'dictionary', 'tuple', 'array', 'set', 'map',
    'object', 'instance',
    # Special values
    'nan', 'inf', 'none', 'null', 'na',
    # Commit prefix typo
    'ehn',
}


# Commit message prefix tags (not code entities)
COMMIT_PREFIXES = {
    'enh', 'fix', 'bug', 'mrg', 'maint', 'tst', 'fea', 'feat',
    'refactor', 'api', 'doc', 'docs', 'wip', 'chore', 'ci',
    'build', 'style', 'perf', 'revert', 'cosmit', 'misc',
    'cleanup', 'typo', 'dep', 'deprecate', 'deprecated',
}

CODE_STOPWORDS = CODE_STOPWORDS | COMMIT_PREFIXES


# Python built-in exception class names
PYTHON_EXCEPTIONS = {
    'exception', 'basexception', 'error',
    'valueerror', 'typeerror', 'keyerror', 'indexerror', 'attributeerror',
    'runtimeerror', 'importerror', 'ioerror', 'oserror', 'nameerror',
    'syntaxerror', 'notimplementederror', 'stopiteration', 'assertionerror',
    'zerodivisionerror', 'overflowerror', 'memoryerror', 'recursionerror',
    'filenotfounderror', 'permissionerror', 'timeouterror', 'connectionerror',
    'unicodeerror', 'unicodedecodeerror', 'unicodeencodeerror',
    'lookuperror', 'arithmeticerror', 'environmenterror',
    'deprecationwarning', 'userwarning', 'futurewarning', 'pendingdeprecationwarning',
}

# Generic abbreviations (too broad; not domain-specific)
GENERIC_ABBREVIATIONS = {
    'func', 'fn', 'cb', 'val', 'var', 'tmp', 'temp', 'obj', 'ref', 'ptr',
    'src', 'dst', 'msg', 'err', 'res', 'ret', 'ctx', 'cfg', 'env', 'def',
}

CODE_STOPWORDS = CODE_STOPWORDS | PYTHON_EXCEPTIONS | GENERIC_ABBREVIATIONS


# Common commit-message verbs (also filter their capitalized forms)
COMMIT_VERBS = {
    # Past tense / past participle
    'added', 'fixed', 'moved', 'improved', 'updated', 'changed', 'removed',
    'implemented', 'optimised', 'optimized', 'adapted', 'merged', 'resolved',
    'corrected', 'cleaned', 'refactored', 'renamed', 'replaced', 'deprecated',
    'introduced', 'simplified', 'extended', 'supported', 'enabled', 'disabled',
    # Present participle
    'adding', 'fixing', 'moving', 'improving', 'updating', 'changing', 'removing',
    'implementing', 'optimising', 'optimizing', 'adapting', 'merging', 'resolving',
    'correcting', 'cleaning', 'refactoring', 'renaming', 'replacing', 'deprecating',
    'introducing', 'simplifying', 'extending', 'supporting', 'enabling', 'disabling',
    # Base form
    'add', 'fix', 'move', 'improve', 'update', 'change', 'remove',
    'implement', 'optimise', 'optimize', 'adapt', 'merge', 'resolve',
    'correct', 'clean', 'refactor', 'rename', 'replace', 'deprecate',
    'introduce', 'simplify', 'extend', 'support', 'enable', 'disable',
    'calculate', 'ensure', 'allow', 'avoid', 'handle', 'include', 'exclude',
    'convert', 'transform', 'apply', 'use', 'using', 'used',
    # Other common words
    'my', 'the', 'this', 'that', 'some', 'all', 'new', 'old',
}

CODE_STOPWORDS = CODE_STOPWORDS | COMMIT_PREFIXES | COMMIT_VERBS

# Commit message prefix tags (not code entities)
COMMIT_PREFIXES = {
    'enh', 'fix', 'bug', 'mrg', 'maint', 'tst', 'fea', 'feat',
    'refactor', 'api', 'doc', 'docs', 'wip', 'chore', 'ci',
    'build', 'style', 'perf', 'revert', 'cosmit', 'misc',
    'cleanup', 'typo', 'dep', 'deprecate', 'deprecated',
    'gp', 'gps',  # often labels; can be confused with Gaussian Process
}

CODE_STOPWORDS = CODE_STOPWORDS | COMMIT_PREFIXES

# Step 1: Lowercase whitelist (common abbreviations in ML/scientific computing)
LOWERCASE_WHITELIST = {
    # ML algorithm abbreviations
    'svm', 'svc', 'svr', 'knn', 'mlp', 'pca', 'lda', 'qda', 'nmf', 'gmm', 'hmm',
    'sgd', 'adam', 'lbfgs', 'bfgs', 'lars', 'lasso', 'ridge', 'enet',
    'rf', 'gbdt', 'xgb', 'lgb', 'catboost',
    'cnn', 'rnn', 'lstm', 'gru', 'gan', 'vae', 'bert', 'gpt',
    # Activations / losses
    'relu', 'tanh', 'sigmoid', 'softmax', 'gelu', 'mse', 'mae', 'rmse',
    # Regularization / normalization
    'l1', 'l2', 'bn', 'ln',
    # Feature / text processing
    'idf', 'tfidf', 'bow', 'ngram', 'word2vec',
    # Clustering / dimensionality reduction
    'tsne', 'umap', 'dbscan', 'kmeans', 'kmedoids',
    # Datasets
    'mnist', 'cifar', 'imagenet', 'coco',
    # Hardware / formats
    'cpu', 'gpu', 'cuda', 'tpu',
    'csv', 'json', 'xml', 'html', 'yaml', 'hdf5', 'parquet',
    # Images
    'rgb', 'bgr', 'hsv', 'jpeg', 'png',
    # Other
    'api', 'url', 'uri', 'http', 'https', 'io', 'cv',
}


# ==================== Logging Setup ====================

def setup_logging(log_folder: str, instance_name: str) -> logging.Logger:
    """Configure logger (file output only; no console output)."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{instance_name}.log")

    logger = logging.getLogger(instance_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False  # Do not propagate to root logger

    # File handler (DEBUG level)
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(fh)
    # No console handler; file-only logging

    return logger


# ==================== Helper Functions ====================

def stem_word(word: str) -> str:
    """Stem a word."""
    if STEMMER is None:
        return word.lower()
    try:
        return STEMMER.stem(word.lower())
    except Exception:
        return word.lower()


# ==================== Step 0: Pre-filtering ====================

def is_low_quality_commit(msg: str, logger: logging.Logger) -> bool:
    """Determine whether a commit message is low quality."""
    if not msg:
        logger.debug("    commit message is empty")
        return True

    msg_stripped = msg.strip()
    if len(msg_stripped) < 10:
        logger.debug(f"    commit message too short: '{msg_stripped}'")
        return True

    # Check if it's only a prefix tag
    for prefix in LOW_QUALITY_PREFIXES:
        # Match: "[MRG+1] MAINT" or "MAINT:" or "MAINT " etc.
        pattern = rf'^\[?(?:MRG\+?\d?)?\]?\s*{prefix}\s*:?\s*(.*)$'
        match = re.match(pattern, msg_stripped, re.IGNORECASE)
        if match:
            remaining = match.group(1).strip()
            if len(remaining) < 5:
                logger.debug(f"    low-quality prefix '{prefix}', remaining too short: '{remaining}'")
                return True

    # Check low-quality patterns
    msg_lower = msg_stripped.lower()
    for pattern in LOW_QUALITY_PATTERNS:
        if re.match(pattern, msg_lower, re.IGNORECASE):
            logger.debug(f"    matched low-quality pattern: '{msg_stripped}'")
            return True

    return False


def compute_tau(co_mod: Dict[str, int], min_tau: int, auto_threshold: bool, logger: logging.Logger) -> int:
    """Compute frequency threshold τ."""
    if not auto_threshold:
        return min_tau

    if not co_mod:
        logger.debug(f"    co_modify is empty, use min_tau={min_tau}")
        return min_tau

    freqs = list(co_mod.values())
    median_freq = statistics.median(freqs)
    tau = max(min_tau, int(median_freq))
    logger.debug(f"    freq distribution: min={min(freqs)}, max={max(freqs)}, median={median_freq:.1f}, τ={tau}")

    return tau


def filter_co_modifications(co_mod: Dict[str, int], tau: int, logger: logging.Logger) -> Dict[str, int]:
    """Filter out low-frequency co-modifications."""
    filtered = {func: freq for func, freq in co_mod.items() if freq >= tau}
    logger.debug(f"    co_modify filtered: {len(co_mod)} -> {len(filtered)} (τ>={tau})")
    return filtered


# ==================== Step 1: Code Entity Extraction ====================

def is_camel_case_multi(token: str) -> bool:
    """Check whether token is multi-word CamelCase (e.g., MiniBatchKMeans)."""
    if not token or len(token) < 3:
        return False
    # At least 2 uppercase letters, not all caps, starts with uppercase
    upper_positions = [i for i, c in enumerate(token) if c.isupper()]
    return (len(upper_positions) >= 2 and
            not token.isupper() and
            token[0].isupper())


def is_upper_abbr(token: str) -> bool:
    """Check whether token is an all-uppercase abbreviation (e.g., SGD, PCA)."""
    return len(token) >= 2 and token.isupper() and token.isalpha()


def is_snake_case(token: str) -> bool:
    """Check whether token is snake_case (e.g., label_propagation)."""
    return ('_' in token and
            not token.startswith('_') and
            not token.endswith('_') and
            len(token) >= 3)


def is_capitalized_single_word(token: str) -> bool:
    """Check whether token is a capitalized single word (e.g., Predict, Token)."""
    if not token or len(token) < 2:
        return False
    return (token[0].isupper() and
            token[1:].islower() and
            token.isalpha())


def is_pure_lowercase(token: str) -> bool:
    """Check whether token is a pure lowercase alphabetic word."""
    return token.islower() and token.isalpha()


def split_camel_case(token: str) -> List[str]:
    """Split CamelCase: MiniBatchKMeans -> [Mini, Batch, KMeans]."""
    parts = re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+(?![a-z])', token)
    return [p for p in parts if len(p) > 0]


def split_snake_case(token: str) -> List[str]:
    """Split snake_case: label_propagation -> [label, propagation]."""
    return [p for p in token.split('_') if len(p) > 0]


def extract_raw_tokens(text: str) -> List[str]:
    """Extract all candidate tokens from text."""
    return re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text)


def classify_and_filter_token(token: str, is_sentence_start: bool, logger: logging.Logger) -> Optional[str]:
    """
    Classify and filter a token.
    Returns: the kept token or None.

    Args:
        token: token to classify
        is_sentence_start: whether it's the first token (treated as sentence start)
        logger: logger instance
    """
    if not token or len(token) < 2:
        return None

    # Filter: too short (<=2) tends to be ambiguous
    if len(token) <= 2:
        logger.debug(f"      filtered (too short): {token}")
        return None

    # Filter: contains year (19xx, 20xx)
    if re.search(r'(?:19|20)\d{2}', token):
        logger.debug(f"      filtered (contains year): {token}")
        return None

    token_lower = token.lower()

    # Stopword filtering
    if token_lower in CODE_STOPWORDS:
        logger.debug(f"      filtered (stopword): {token}")
        return None

    # Keep: multi-word CamelCase
    if is_camel_case_multi(token):
        logger.debug(f"      kept (CamelCase): {token}")
        return token

    # Keep: all-uppercase abbreviations (must be whitelisted or not an English word)
    if is_upper_abbr(token):
        if token_lower in LOWERCASE_WHITELIST:
            logger.debug(f"      kept (UPPER_ABBR+whitelist): {token}")
            return token
        elif ENGLISH_WORDS and token_lower in ENGLISH_WORDS:
            logger.debug(f"      filtered (UPPER_ABBR but English word): {token}")
            return None
        else:
            logger.debug(f"      kept (UPPER_ABBR): {token}")
            return token

    # Keep: snake_case
    if is_snake_case(token):
        logger.debug(f"      kept (snake_case): {token}")
        return token

    # Keep: whitelisted lowercase tokens
    if token_lower in LOWERCASE_WHITELIST:
        logger.debug(f"      kept (whitelist): {token}")
        return token

    # Filter: capitalized single word at sentence start (might just be English sentence capitalization)
    if is_sentence_start and is_capitalized_single_word(token):
        logger.debug(f"      filtered (sentence-start capitalized word): {token}")
        return None

    # Capitalized single word: filter if it's an English word (including inflections), else keep
    if is_capitalized_single_word(token):
        if ENGLISH_WORDS:
            token_stem = stem_word(token)
            if token_lower in ENGLISH_WORDS or token_stem in ENGLISH_WORD_STEMS:
                logger.debug(f"      filtered (English word): {token}")
                return None
        logger.debug(f"      kept (capitalized single word): {token}")
        return token

    # Filter: non-whitelisted pure lowercase words
    # Optimization: keep long non-English lowercase words (may be domain-specific, e.g., elasticsearch)
    if is_pure_lowercase(token):
        # Length threshold: longer lowercase words are more likely to be meaningful domain words
        MIN_LOWERCASE_LENGTH = 8

        if len(token) >= MIN_LOWERCASE_LENGTH:
            # Check whether it's an English word (including inflections)
            token_stem = stem_word(token)
            if ENGLISH_WORDS and (token in ENGLISH_WORDS or token_stem in ENGLISH_WORD_STEMS):
                logger.debug(f"      filtered (long lowercase but English word): {token}")
                return None
            else:
                # Long lowercase non-English word could be domain-specific
                logger.debug(f"      kept (long lowercase non-English word): {token}")
                return token
        else:
            logger.debug(f"      filtered (short lowercase): {token}")
            return None

    # Filter: contains digits (e.g., v2, test123)
    if any(c.isdigit() for c in token):
        logger.debug(f"      filtered (contains digits): {token}")
        return None

    # Default: filter other cases
    logger.debug(f"      filtered (other): {token}")
    return None


def extract_code_entities(text: str, source_label: str, logger: logging.Logger) -> List[str]:
    """Extract code entities from text."""
    logger.debug(f"    extracting from [{source_label}]: '{text[:80]}{'...' if len(text) > 80 else ''}'")

    raw_tokens = extract_raw_tokens(text)
    entities = []

    for idx, token in enumerate(raw_tokens):
        is_sentence_start = (idx == 0)  # Treat the first token as sentence start
        filtered = classify_and_filter_token(token, is_sentence_start, logger)
        if filtered:
            entities.append(filtered)

    logger.debug(f"    [{source_label}] extracted {len(entities)} entities: {entities[:10]}{'...' if len(entities) > 10 else ''}")
    return entities


# ==================== Step 2: Normalization + Deduplication ====================

def normalize_token_for_dedup(token: str) -> Set[str]:
    """
    Normalize a token for dedup comparisons.
    Returns all normalized forms for matching.
    """
    normalized_forms = set()

    # Split token
    if is_camel_case_multi(token):
        parts = split_camel_case(token)
    elif is_snake_case(token):
        parts = split_snake_case(token)
    else:
        parts = [token]

    # Add stemmed forms for each part
    for part in parts:
        normalized_forms.add(stem_word(part))

    # Add sorted-joined form (so MiniBatchKMeans matches mini_batch_kmeans)
    sorted_stems = sorted([stem_word(p) for p in parts])
    normalized_forms.add('_'.join(sorted_stems))

    return normalized_forms


def get_test_func_normalized_tokens(test_func: str) -> Set[str]:
    """Get all normalized tokens for a test_func."""
    normalized = set()
    raw_tokens = extract_raw_tokens(test_func)

    for token in raw_tokens:
        normalized.update(normalize_token_for_dedup(token))

    return normalized


def filter_overlapping_tokens(
    candidate_tokens: List[str],
    test_func: str,
    logger: logging.Logger
) -> List[str]:
    """
    Filter tokens that overlap with test_func tokens.

    Fix: for CamelCase / uppercase abbreviations, use whole-token matching rather than split-part matching.
    Rationale: XMLWriter as a complete class name should be kept even if test_xml_writer contains 'xml' and 'writer'.
    """
    # Lowercase test_func for whole-token substring matching
    test_func_lower = test_func.lower()

    # Tokenize test_func for non-CamelCase matching
    test_raw_tokens = extract_raw_tokens(test_func)
    test_tokens_lower = set(t.lower() for t in test_raw_tokens)

    logger.debug(f"  test_func_lower: {test_func_lower}")
    logger.debug(f"  test_tokens_lower ({len(test_tokens_lower)}): {sorted(list(test_tokens_lower))[:20]}...")

    filtered = []
    for token in candidate_tokens:
        # Strategy 1: CamelCase and UPPER abbreviations -> whole-token substring match only
        if is_camel_case_multi(token) or is_upper_abbr(token):
            token_lower = token.lower()
            # Only filter if the whole token appears as a substring in test_func
            if token_lower in test_func_lower:
                logger.debug(f"    filtered (whole overlap): {token}")
                continue
            else:
                filtered.append(token)
                logger.debug(f"    kept (Camel/abbr, no whole overlap): {token}")

        # Strategy 2: snake_case -> filter only if ALL parts overlap
        elif is_snake_case(token):
            parts = split_snake_case(token)
            parts_lower = set(p.lower() for p in parts)
            new_parts = parts_lower - test_tokens_lower

            if not new_parts:
                logger.debug(f"    filtered (snake_case all overlap): {token}")
                continue
            else:
                filtered.append(token)
                logger.debug(f"    kept (snake_case has new info): {token}, new parts: {new_parts}")

        # Strategy 3: other single tokens -> direct appearance check
        else:
            token_lower = token.lower()
            if token_lower in test_tokens_lower:
                logger.debug(f"    filtered (word overlap): {token}")
                continue
            else:
                filtered.append(token)
                logger.debug(f"    kept (word no overlap): {token}")

    return filtered

# ==================== Main Processing Logic ====================

def get_co_modification_dict(test_func: str, item: dict) -> Dict[str, int]:
    """Extract co_modification dict."""
    co_modifications = item.get("co_modifications", None)
    co_mod_dict = {}

    if co_modifications:
        for commit in co_modifications:
            modified_entities = commit.get("modified_entities", None)
            if modified_entities:
                for entity in modified_entities:
                    if entity == test_func:
                        continue
                    co_mod_dict[entity] = co_mod_dict.get(entity, 0) + 1

    return co_mod_dict


def process_single_test(
    test_func: str,
    item: dict,
    min_tau: int,
    auto_threshold: bool,
    logger: logging.Logger
) -> Dict:
    """Process a single test function."""
    logger.debug(f"Processing test function: {test_func}")

    all_candidate_tokens = []

    # ========== Process init_commit ==========
    init_commit = item.get("init_commit", None)
    if init_commit:
        commit_msg = init_commit.get("commit_message", "")
        if not is_low_quality_commit(commit_msg, logger):
            tokens = extract_code_entities(commit_msg, "init_commit", logger)
            all_candidate_tokens.extend(tokens)
        else:
            logger.debug("  init_commit filtered out (low quality)")
    else:
        logger.debug("  No init_commit")

    # ========== Process co_modifications ==========
    # Diagnostic logs: check the raw data structure
    co_modifications_raw = item.get("co_modifications", None)
    if co_modifications_raw is None:
        logger.warning("  co_modifications field is missing")
    elif len(co_modifications_raw) == 0:
        logger.warning("  co_modifications is an empty list")
    else:
        logger.info(f"  co_modifications raw record count: {len(co_modifications_raw)}")
        # Check the structure of the first record
        if co_modifications_raw:
            first_record = co_modifications_raw[0]
            logger.debug(f"  first co_modifications record keys: {list(first_record.keys())}")
            modified_entities = first_record.get("modified_entities", [])
            logger.debug(f"  first record modified_entities count: {len(modified_entities)}")

    co_mod_dict = get_co_modification_dict(test_func, item)
    logger.debug(f"  co_modify raw entry count: {len(co_mod_dict)}")

    if co_mod_dict:
        # Compute τ
        tau = compute_tau(co_mod_dict, min_tau, auto_threshold, logger)

        # Frequency filtering
        filtered_co_mod = filter_co_modifications(co_mod_dict, tau, logger)

        # Extract tokens from entity names
        for entity in filtered_co_mod.keys():
            tokens = extract_code_entities(entity, "co_modify", logger)
            all_candidate_tokens.extend(tokens)
    else:
        logger.debug("  No co_modify data")

    # ========== Deduplicate (token-level) ==========
    seen = set()
    unique_tokens = []
    for t in all_candidate_tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    logger.debug(f"  Candidate tokens dedup: {len(all_candidate_tokens)} -> {len(unique_tokens)}")

    # ========== Deduplicate vs test_func ==========
    final_tokens = filter_overlapping_tokens(unique_tokens, test_func, logger)

    logger.info(f"  [{test_func.split('::')[-1]}] final tokens: {len(final_tokens)}")
    if final_tokens:
        logger.debug(f"    result: {final_tokens}")

    return {
        "test_func": test_func,
        "domain_knowledge_tokens": final_tokens
    }


def process_instance(
    input_file: str,
    output_file: str,
    min_tau: int,
    auto_threshold: bool,
    logger: logging.Logger
) -> None:
    """Process a single instance file."""
    logger.info(f"Reading input file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Total test functions to process: {len(data)}")
    logger.info(f"Parameters: min_tau={min_tau}, auto_threshold={auto_threshold}")

    results = []

    for idx, (test_func, item) in enumerate(data.items()):
        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(data)}")

        result = process_single_test(test_func, item, min_tau, auto_threshold, logger)
        results.append(result)

    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved: {output_file}")

    # Statistics
    total_tokens = sum(len(r["domain_knowledge_tokens"]) for r in results)
    non_empty_count = sum(1 for r in results if r["domain_knowledge_tokens"])
    avg_tokens = total_tokens / len(results) if results else 0

    logger.info("=" * 50)
    logger.info("Statistics:")
    logger.info(f"  Total tests: {len(results)}")
    logger.info(f"  Tests with domain knowledge: {non_empty_count} ({non_empty_count/len(results)*100:.1f}%)")
    logger.info(f"  Total token count: {total_tokens}")
    logger.info(f"  Avg tokens per test: {avg_tokens:.2f}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Domain Knowledge Enhancement: Extract domain knowledge tokens from historical information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python domain_knowledge_enhancement.py \\
    --raw_historical_info_folder /path/to/historical_information \\
    --output_folder /path/to/output \\
    --log_folder /path/to/logs \\
    --min_tau 3 \\
    --auto_thresholds
        """
    )

    parser.add_argument(
        "--raw_historical_info_folder",
        type=str,
        required=True,
        help="Folder containing raw historical info JSON files"
    )
    parser.add_argument(
        "--min_tau",
        type=int,
        default=3,
        help="Minimum frequency threshold for co-modification filtering (default: 3)"
    )
    parser.add_argument(
        "--auto_thresholds",
        action="store_true",
        help="If set, τ = max(min_tau, median(freq)); otherwise τ = min_tau"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save output JSON files"
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        required=True,
        help="Folder to save log files"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.log_folder, exist_ok=True)

    # Gather all input files
    input_folder = Path(args.raw_historical_info_folder)
    input_files = sorted(input_folder.glob("*.json"))

    if not input_files:
        print(f"Error: No JSON files found in {args.raw_historical_info_folder}")
        return

    print(f"Found {len(input_files)} input files")
    print(f"Output folder: {args.output_folder}")
    print(f"Log folder: {args.log_folder}")
    print("=" * 60)

    # Process each instance
    for i, input_file in enumerate(input_files):
        instance_name = input_file.stem
        output_file = os.path.join(args.output_folder, f"{instance_name}.json")

        print(f"\n[{i+1}/{len(input_files)}] Processing instance: {instance_name}")

        # Setup logger
        logger = setup_logging(args.log_folder, instance_name)
        logger.info("=" * 60)
        logger.info(f"Instance: {instance_name}")
        logger.info("=" * 60)

        try:
            process_instance(
                str(input_file),
                output_file,
                args.min_tau,
                args.auto_thresholds,
                logger
            )
            print(f"    Done: {output_file}")
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            print(f"    Failed: {e}")

        # Close logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    print("\n" + "=" * 60)
    print("All processing completed")


if __name__ == "__main__":
    main()

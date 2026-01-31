import json
import re
import warnings
import logging

import editdistance
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from codebleu import calc_codebleu  # official CodeBLEU library


PYTHON_COMMENT_PATTERN = re.compile(r"#.*$")  # match Python comments
DIFF_META_PATTERN = re.compile(r"^(diff --git|index |--- |\+\+\+ |@@.*@@)")  # match diff meta information lines


def preprocess_diff(diff_text: str) -> str:
    """
    preprocess diff patch: extract valid code modification lines, filter out irrelevant meta information
    keep: +/- prefixed code lines (remove symbols), non-meta information code lines
    filter: diff meta information, comments, empty lines
    """
    if not diff_text:
        return ""

    lines = diff_text.strip().split("\n")
    valid_lines = []

    for line in lines:
        line = line.strip()
        # 1. filter diff meta information lines (file path, line number markers, etc.)
        if DIFF_META_PATTERN.match(line):
            continue
        # 2. filter empty lines
        if not line:
            continue
        # 3. filter Python comment lines (single comment lines)
        if line.startswith("#"):
            continue
        # 4. process +/- prefixed modification lines (remove symbols, keep code)
        if line.startswith(("+", "-")):
            code_line = line[1:].strip()  # remove +/- symbols
        else:
            code_line = line

        # 5. filter inline comments (keep code part)
        code_line = PYTHON_COMMENT_PATTERN.sub("", code_line).strip()
        if code_line:  # filter empty lines after processing
            valid_lines.append(code_line)

    # join valid code lines, preserve order (code order affects logic)
    return "\n".join(valid_lines)


def python_code_tokenize(code: str) -> list:
    """
    Python code tokenization: keep variable names, function names, keywords, symbols, compatible with Python syntax
    e.g. "def _get_rows(item):" → ["def", "_get_rows", "(", "item", ")"]
    """
    if not code:
        return []

    # match Python identifiers (letters, numbers, underscores), symbols, keywords
    # regex rule: match identifiers + match non-alphanumeric symbols (keep parentheses, commas, colons, etc.)
    token_pattern = r"[a-zA-Z0-9_]+|[()\[\]{}:;,.+*/=<>!&|%^~@#-]"
    tokens = re.findall(token_pattern, code)

    # filter empty tokens and pure space tokens
    tokens = [token.strip() for token in tokens if token.strip()]

    # optional: filter Python keywords (if not needed to distinguish keywords from ordinary identifiers)
    # python_keywords = {"def", "class", "if", "else", "return", "for", "while", ...}
    # tokens = [token for token in tokens if token not in python_keywords]

    return tokens


def jaccard_similarity(s1_tokens: list, s2_tokens: list) -> float:
    """Jaccard similarity based on tokens (reflects code element overlap)"""
    if not s1_tokens and not s2_tokens:
        return 1.0
    if not s1_tokens or not s2_tokens:
        return 0.0
    set1 = set(s1_tokens)
    set2 = set(s2_tokens)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return round(intersection / union if union != 0 else 0.0, 4)


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized edit distance (0=completely same, 1=completely different), based on preprocessed code text"""
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    dist = editdistance.eval(s1, s2)
    return round(dist / max_len, 4)


def tfidf_cosine_similarity(s1: str, s2: str, vectorizer: TfidfVectorizer) -> float:
    """TF-IDF+cosine similarity (based on Python code tokenization)"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    tfidf_matrix = vectorizer.transform([s1, s2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity), 4)


def calculate_codebleu(ground_truth_code: str, prediction_code: str) -> float:
    """Calculate CodeBLEU (Python code-specific, optimized weights)"""
    if not ground_truth_code and not prediction_code:
        return 1.0
    if not ground_truth_code or not prediction_code:
        return 0.0

    references = [[ground_truth_code]]  # 2D list (compatible with multiple references)
    predictions = [prediction_code]  # must be a 1D list (official requirement)

    # weights meaning (official definition): (ngram_match, weighted_ngram_match, syntax_match, dataflow_match)
    # your requirement: prioritize 1-2gram (first two parameters) and AST (syntax_match, third parameter)
    # suppress codebleu library about data-flow warnings (when data-flow cannot be extracted, weights are set to 0.0, does not affect the result)
    # codebleu library uses logging instead of warnings, need to temporarily set root logger's log level
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        # temporarily suppress WARNING level logs (but does not affect ERROR and higher levels)
        root_logger.setLevel(logging.ERROR)
        codebleu_result = calc_codebleu(
            references=references,  # reference text (2D list)
            predictions=predictions,  # prediction text (1D list)
            lang="python",  # language (required)
            weights=(0.4, 0.4, 0.2, 0.0)  # adjust weights according to your requirement
        )
    finally:
        # restore original log level
        root_logger.setLevel(original_level)

    # 3. extract CodeBLEU total score from the returned dictionary (official key is "codebleu")
    codebleu_score = codebleu_result.get("codebleu", 0.0)
    return round(codebleu_score, 4)



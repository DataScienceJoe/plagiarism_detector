import os
import numpy as np
import pandas as pd
import re
from collections import Counter

def read_text_file(file_path):
    """Reads a text file, removes unwanted characters, and returns cleaned text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation except spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing
    return text

def tokenize(text):
    """Splits text into words, ignoring punctuation and extra spaces."""
    return [word for word in re.findall(r'\b\w+\b', text)]

def get_ngrams(words, n):
    return Counter([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])

def ngram_overlap(text1, text2, n=10):
    words1, words2 = tokenize(text1), tokenize(text2)
    ngrams1, ngrams2 = get_ngrams(words1, n), get_ngrams(words2, n)
    common_keys = set(ngrams1.keys()) & set(ngrams2.keys())
    intersection = sum(min(ngrams1[key], ngrams2[key]) for key in common_keys)
    union = sum(ngrams1.values()) + sum(ngrams2.values()) - intersection
    return intersection / union if union != 0 else 0

def longest_common_subsequence(text1, text2):
    words1, words2 = tokenize(text1), tokenize(text2)
    len1, len2 = len(words1), len(words2)
    if len1 == 0 or len2 == 0:
        return 0
    prev = np.zeros(len2 + 1, dtype=int)
    curr = np.zeros(len2 + 1, dtype=int)
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len2] / max(len1, len2)

def compare_text_files(file1, file2):
    text1, text2 = read_text_file(file1), read_text_file(file2)
    return {
        "Longest Common Subsequence": longest_common_subsequence(text1, text2),
        "N-gram Overlap (n=3)": ngram_overlap(text1, text2)
    }

# Example usage
if __name__ == "__main__":
    file1 = "data/bill_of_rights2.txt"
    file2 = "data/bill_of_rights4.txt"
    if os.path.exists(file1) and os.path.exists(file2):
        results = compare_text_files(file1, file2)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("Ensure both files exist before running the script.")

import os
import numpy as np
import pandas as pd


def read_text_file(file_path):
    """Reads a text file and returns a cleaned list of words."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()  # Convert to lowercase for normalization
    return text

def tokenize(text):
    """Splits text into words, ignoring punctuation and extra spaces."""
    return [word for word in text.split() if word.isalnum()]

def longest_common_subsequence(text1, text2):
    """Finds the length of the longest common subsequence (LCS) between two texts."""
    words1, words2 = tokenize(text1), tokenize(text2)
    len1, len2 = len(words1), len(words2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[len1][len2] / max(len1, len2) if max(len1, len2) != 0 else 0


def ngram_overlap(text1, text2, n=3):
    """Computes the overlap of n-grams between two texts."""

    def get_ngrams(words, n):
        return set([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])

    words1, words2 = tokenize(text1), tokenize(text2)
    ngrams1, ngrams2 = get_ngrams(words1, n), get_ngrams(words2, n)
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union != 0 else 0


def compare_text_files(file1, file2):
    """Compares two text files and prints similarity measures."""
    text1, text2 = read_text_file(file1), read_text_file(file2)

    lcs = longest_common_subsequence(text1, text2)
    ngram = ngram_overlap(text1, text2)

    return {
        "Longest Common Subsequence": lcs,
        "N-gram Overlap (n=3)": ngram
    }


# Example usage
if __name__ == "__main__":
    file1 = "data/bill_of_rights1.txt"
    file2 = "data/bill_of_rights2.txt"
    if os.path.exists(file1) and os.path.exists(file2):
        results = compare_text_files(file1, file2)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("Ensure both files exist before running the script.")

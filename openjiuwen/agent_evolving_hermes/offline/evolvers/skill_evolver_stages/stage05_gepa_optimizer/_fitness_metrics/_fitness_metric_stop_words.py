# Common English stop words — filtered out during extraction
from typing import Set


STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "than", "that", "this", "these", "those", "it", "its", "also",
    "if", "then", "when", "where", "which", "who", "what", "how",
    "all", "any", "some", "such", "more", "most", "other", "same",
    "just", "about", "up", "out", "into", "through", "during",
}

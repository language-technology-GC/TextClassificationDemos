#!/usr/bin/env python
"""Converts data into Vowpal Wabbit format."""

import argparse
import random
import re

from typing import List, Tuple

import detectormorse  # type: ignore
import nltk           # type: ignore


def main(args: argparse.Namespace) -> None:
    stopwords = frozenset(nltk.corpus.stopwords.words("english"))
    detector = detectormorse.detector.default_model()
    random.seed(args.seed)
    with open(args.input, "r") as source:
        lines: List[Tuple[int, str]] = []
        for line in source:
            document, label = line.rstrip().split("\t", 1)
            int_label = 1 if label == "1" else -1
            doc_tokens = []
            for sentence in detector.segments(document):
                # Tokenizes and case-folds.
                tokens = [
                    token.casefold() for token in nltk.word_tokenize(sentence)
                ]
                # Filters stopwords and non-alphabetic tokens.
                tokens = [
                    token
                    for token in tokens
                    if token not in stopwords
                    and re.match(r"^[a-z\-]+$", token)
                ]
                doc_tokens.extend(tokens)
            document = " ".join(doc_tokens)
            lines.append((int_label, document))
    random.shuffle(lines)
    with open(args.output, "w") as sink:
        for int_label, document in lines:
            print(f"{int_label} | {document}", file=sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("input", help="input file")
    parser.add_argument("output", help="output file")
    main(parser.parse_args())

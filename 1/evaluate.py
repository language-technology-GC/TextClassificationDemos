#!/usr/bin/env python
"""Computes accuracy for Vowpal Wabbit classification problems."""

import argparse


def main(args: argparse.Namespace) -> None:
    correct = 0
    examples = 0
    with open(args.gold, "r") as gold, open(args.hypo, "r") as hypo:
        for gold_line, hypo_line in zip(gold, hypo):
            gold_label = gold_line.split()[0]
            hypo_line = hypo_line.rstrip()
            if gold_label == hypo_line:
                correct += 1
            examples += 1
    print(f"Accuracy:\t{correct / examples:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", help="path to gold fasttext file")
    parser.add_argument("hypo", help="path to hypothesis fasttext file")
    main(parser.parse_args())

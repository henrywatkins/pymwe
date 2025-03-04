"""Command line interface for pymwe."""

import csv
import json
from typing import List, TextIO

import click

from pymwe.model import cfeatures, find_mwe


@click.command()
@click.argument("filename", required=True, type=click.File("r"))
@click.option("-n", default=10, help="Number of MWEs to find.")
@click.option("--min_df", default=5, help="Minimum document frequency.")
@click.option("--max_df", default=0.9, help="Maximum document frequency.")
def main(filename: TextIO, n: int, min_df: int, max_df: float) -> None:
    """Find multi-word expressions in text.

    Reads from a file and extracts the most significant multi-word expressions
    based on statistical measures.

    Args:
        filename: Input file containing text, one document per line
        n: Number of MWEs to find
        min_df: Minimum document frequency for terms to be considered
        max_df: Maximum document frequency for terms to be considered
    """
    if min_df < 1:
        raise ValueError("min_df must be at least 1")

    if max_df > 1.0 or max_df <= 0.0:
        raise ValueError("max_df must be between 0 and 1")

    texts = filename.readlines()
    mwe = find_mwe(texts, n=n, min_df=min_df, max_df=max_df)
    for m in mwe:
        print(m)


@click.command()
@click.argument("groups_file", required=True, type=click.File("r"))
@click.argument("features_file", required=True, type=click.File("r"))
@click.option("--top_k", default=5, help="Number of top features to return for each group.")
@click.option("--show_values", is_flag=True, help="Show correlation values alongside features.")
@click.option("--format", type=click.Choice(["text", "json"]), default="text", 
              help="Output format (text or JSON).")
def cfeatures_cli(
    groups_file: TextIO, 
    features_file: TextIO, 
    top_k: int, 
    show_values: bool,
    format: str
) -> None:
    """Find features most correlated with each group/cluster.

    Calculates Matthews correlation coefficients between features and group IDs
    to identify the most characteristic features of each group.

    Args:
        groups_file: File with group/cluster IDs, one per line
        features_file: CSV file with features, one sample per line
        top_k: Number of top features to return for each group
        show_values: Whether to show correlation values alongside features
        format: Output format (text or JSON)
    """
    try:
        group_ids = [int(line.strip()) for line in groups_file.readlines() if line.strip()]
    except ValueError:
        raise ValueError("Group IDs must be integers")

    try:
        reader = csv.reader(features_file)
        data = [row for row in reader if row]
    except Exception as e:
        raise ValueError(f"Error reading features file: {e}")

    if len(group_ids) != len(data):
        raise ValueError(
            f"Number of group IDs ({len(group_ids)}) must match number of feature rows ({len(data)})"
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    results = cfeatures(group_ids, data, top_k=top_k, show_values=show_values)
    
    if format == "json":
        print(json.dumps(results, default=str))
    else:
        for group_id, features in results.items():
            print(f"Group {group_id}:")
            for feature in features:
                if show_values:
                    feature_name, value = feature
                    print(f"  {feature_name}: {value:.4f}")
                else:
                    print(f"  {feature}")
            print()

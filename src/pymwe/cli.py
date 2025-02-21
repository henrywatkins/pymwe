import click
from pymwe.model import find_mwe


@click.command()
@click.argument("filename", required=True, type=click.File("r"))
@click.option("-n", default=10, help="Number of MWEs to find.")
@click.option("--min_df", default=5, help="Minimum document frequency.")
@click.option("--max_df", default=0.9, help="Maximum document frequency.")
def main(filename, n, min_df, max_df):
    """Reads from a file or standard input and prints its content."""
    texts = filename.readlines()
    mwe = find_mwe(texts, n=n, min_df=min_df, max_df=max_df)
    for m in mwe:
        print(m)

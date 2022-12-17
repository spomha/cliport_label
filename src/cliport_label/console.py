"""CLI entrypoints for cliport_label"""
from importlib import import_module

import click

from cliport_label import __version__
from cliport_label.main import main



@click.command()
@click.version_option(version=__version__)
@click.option("-l", "--loglevel", help="Python log level, 10=DEBUG, 20=INFO, 30=WARNING, 40=CRITICAL", default=30)
@click.option("-v", "--verbose", count=True, help="Shorthand for info/debug loglevel (-v/-vv)")
def cliport_label_cli(loglevel: int, verbose: int) -> None:
    """GUI application for recording demonstration for CLIPort project """
    main()


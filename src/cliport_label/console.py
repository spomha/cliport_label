"""CLI entrypoints for cliport_label"""
from pathlib import Path

import click
import toml

from cliport_label import __version__
from cliport_label.main import main



@click.command()
@click.version_option(version=__version__)
@click.option("-l", "--loglevel", help="Python log level, 10=DEBUG, 20=INFO, 30=WARNING, 40=CRITICAL", default=30)
@click.option("-v", "--verbose", count=True, help="Shorthand for info/debug loglevel (-v/-vv)")
@click.argument("config-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def cliport_label_cli(config_path: Path, loglevel: int, verbose: int,) -> None:
    """GUI application for recording demonstration for CLIPort project """
    config = toml.load(config_path)
    click.echo("GUI Controls")
    for key, value in config['controlkeys'].items():
        click.echo("Press " + click.style(value, fg='green', bold=True) + " to " + click.style(key, fg='white', bold=True))
    main(config)


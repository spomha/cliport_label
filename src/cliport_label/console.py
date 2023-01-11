"""CLI entrypoints for cliport_label"""
from pathlib import Path

import click
import toml

from cliport_label import __version__
from cliport_label.main import main_tool, main_viewer, main_editor



@click.group()
@click.version_option(version=__version__)
@click.option("-l", "--loglevel", help="Python log level, 10=DEBUG, 20=INFO, 30=WARNING, 40=CRITICAL", default=30)
@click.option("-v", "--verbose", count=True, help="Shorthand for info/debug loglevel (-v/-vv)")
@click.argument("config-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.pass_context
def cliport_label_cli(ctx, config_path: Path, loglevel: int, verbose: int,) -> None:
    """GUI application for recording demonstration for CLIPort project """
    ctx.ensure_object(dict)
    config = toml.load(config_path)
    ctx.obj['CONFIG'] = config


@cliport_label_cli.command()
@click.pass_context
def tool(ctx):
    """Run label tool"""
    config = ctx.obj['CONFIG']
    click.echo("GUI Controls")
    for key, value in config['tool_controlkeys'].items():
        click.echo("Press " + click.style(value, fg='green', bold=True) + " to " + click.style(key, fg='white', bold=True))
    main_tool(config)

@cliport_label_cli.command()
@click.pass_context
def viewer(ctx):
    """Run data viewer"""
    config = ctx.obj['CONFIG']

    main_viewer(config)


@cliport_label_cli.command()
@click.pass_context
@click.argument("filepath", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("-l", "--lang-goal", type=str, default="", help="Specify new language goal to set")
def editor(ctx, filepath: Path, lang_goal: str):
    """Run data editor"""
    click.echo(f"Editing file: {filepath}")
    main_editor(filepath, lang_goal)


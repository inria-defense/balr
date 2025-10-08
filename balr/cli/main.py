import logging

import typer
from rich.logging import RichHandler

from balr.binary_attributes.task import app as ba_app
from balr.embeddings.task import app as embeddings_app
from balr.scoring.task import app as scoring_app

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)

app = typer.Typer(no_args_is_help=True)

app.add_typer(embeddings_app)
app.add_typer(ba_app)
app.add_typer(scoring_app, name="score", help="Scoring related commands.")

if __name__ == "__main__":
    app()

from pathlib import Path

from ml.scripts.cli import cli_main as ml_cli_main

PROJECT_ROOT = Path(__file__).parent.parent


def cli_main() -> None:
    ml_cli_main(PROJECT_ROOT)


if __name__ == "__main__":
    cli_main()

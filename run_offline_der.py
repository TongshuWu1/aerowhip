"""Launch the offline DDER application, or its explicit CLI subcommands."""

import sys


def main() -> None:
    if len(sys.argv) == 1:
        from offline_der.gui import main as gui_main

        gui_main()
        return
    from offline_der.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()

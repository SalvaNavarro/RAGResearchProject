import argparse
import runpy


def main(module) -> None:
    """
    High level runner for modules on production mode.
    For different configurations please refer to __main__.py files on 
    the relevant modules.

    Args:
        module (str): THe chose module to run.

    Returns:
        None

    """

    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":

    # arguments for running configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Select module to run.", 
                        default="app")
    args = parser.parse_args()

    main(args.module)

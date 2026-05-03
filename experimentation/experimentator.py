from pathlib import Path
import sys

from experimentation.robustness_runner import (
    RobustnessSuiteRunner,
    build_runner_from_config_path,
    main as robustness_runner_main,
    parse_config,
)


def run_experiments_from_config(config_path):
    runner = build_runner_from_config_path(Path(config_path).resolve())
    return runner.run()


def build_runner(config_path) -> RobustnessSuiteRunner:
    return RobustnessSuiteRunner(parse_config(Path(config_path).resolve()))


def main(argv=None):
    robustness_runner_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])

# Social Laws in the Unified Planning Framework

This repository contains the Unified Planning Framework social-laws engine and
the experiment harness used to evaluate robustness verifiers for multi-agent
planning problems.


## Installation

Install the package in editable mode when working from this repository:

```bash
pip install -e .
```

The released engine can also be installed as an extra requirement of the
`unified-planning` package:

```bash
pip install unified-planning[social-law]
```


## Experimentation CLI

Use the unified experimentation entrypoint for full runs, validation checks,
single-case debugging, interactive simulation, and analysis.

Run the active experiment suite:

```bash
python -m experimentation.cli run experimentation/all_domains_tests.json
```

Resume an interrupted suite in the same run directory:

```bash
python -m experimentation.cli run experimentation/all_domains_tests.json --resume
```

Check that configured cases load and compile:

```bash
python -m experimentation.cli loading-check experimentation/all_domains_tests.json
```

Populate the compiled robustness PDDL cache during the loading check:

```bash
python -m experimentation.cli loading-check experimentation/all_domains_tests.json --write-pddl-cache
```

Reuse cached compiled robustness PDDL during a full run:

```bash
python -m experimentation.cli run experimentation/all_domains_tests.json --use-pddl-cache
```

Run the suite and validate expected robustness outcomes:

```bash
python -m experimentation.cli expected-check experimentation/all_domains_tests.json
```

Debug one case/verifier pair:

```bash
python -m experimentation.cli single-case numeric_zenotravel pfile1.json simple
python -m experimentation.cli single-case numeric_zenotravel pfile1.json general --with-sl
```

Interactively simulate a compiled robustness problem:

```bash
python -m experimentation.cli simulate numeric_zenotravel pfile1.json --verifier simple
python -m experimentation.cli simulate numeric_zenotravel pfile1.json --with-sl --verifier general --show-state
```

Analyze a completed run directory:

```bash
python -m experimentation.cli analyze experimentation/runs/all_domains_tests
```

`experimentation/all_domains_tests.json` is the full active experiment config.
It writes run artifacts under `experimentation/runs/`, uses the `enhsp` engine by
default, and currently runs both the `general` and `simple` verifiers. When
`--write-pddl-cache` is enabled, compiled robustness problems are written under
`experimentation/runs/<run_id>/compiled_pddl/` as `domain.pddl`, `problem.pddl`,
and `metadata.json` files.

The config intentionally excludes `numeric_civ` and `numeric_grid`: `numeric_civ`
is currently broken/unstable, and `numeric_grid` is not genuinely numeric after
the simple compilation because its coordinate fluents are redundant bookkeeping.


## Acknowledgments
<img src="https://www.aiplan4eu-project.eu/wp-content/uploads/2021/07/euflag.png" width="60" height="40">

This library is being developed for the AIPlan4EU H2020 project (https://aiplan4eu-project.eu) that is funded by the European Commission under grant agreement number 101016442.

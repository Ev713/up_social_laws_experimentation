import argparse
import contextlib
import csv
import json
import resource
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import StringIO
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from unified_planning.engines import CompilationKind
from unified_planning.engines.results import (
    POSITIVE_OUTCOMES,
    PlanGenerationResultStatus,
)
from unified_planning.shortcuts import Compiler, OneshotPlanner, get_environment

from experimentation.problem_generators.blocksworld_generator import BlocksworldGenerator
from experimentation.problem_generators.driverlog_generator import DriverLogGenerator
from experimentation.problem_generators.expedition_generator import ExpeditionGenerator
from experimentation.problem_generators.grid_generator import GridGenerator
from experimentation.problem_generators.intersection_problem_generator import IntersectionProblemGenerator
from experimentation.problem_generators.market_trader_generator import MarketTraderGenerator
from experimentation.problem_generators.numeric_civ_generator import NumericCivGenerator
from experimentation.problem_generators.numeric_grid_generator import NumericGridGenerator
from experimentation.problem_generators.numeric_intersection_generator import NumericIntersectionGenerator
from experimentation.problem_generators.numeric_zenotravel_generator import NumericZenotravelGenerator
from experimentation.problem_generators.zenotravel_generator import ZenoTravelGenerator
from up_social_laws.robustness_checker import SocialLawRobustnessStatus
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.snp_to_num_strips import MultiAgentWithWaitforNumericStripsProblemConverter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experimentation" / "runs"
get_environment().credits_stream = None
warnings.filterwarnings(
    "ignore",
    message="We cannot establish whether .* can solve this problem!",
)


class ProgressReporter:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8")

    def close(self):
        self._handle.close()

    def emit(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()


@dataclass(frozen=True)
class DomainSpec:
    name: str
    generator_cls: type
    instances_dir: Path
    supports_social_law: bool


@dataclass(frozen=True)
class VerifierSpec:
    label: str
    compiler_name: str
    requires_snp: bool


@dataclass
class ResourceLimits:
    engine: str = "enhsp"
    planner_timeout_seconds: Optional[int] = None
    wall_timeout_seconds: int = 1800
    cpu_seconds: int = 1800
    memory_bytes: int = 16_000_000_000


@dataclass(frozen=True)
class ProblemCase:
    domain: str
    instance_file: str
    has_social_law: bool

    @property
    def social_law_label(self) -> str:
        return "with_sl" if self.has_social_law else "without_sl"

    @property
    def case_id(self) -> str:
        return f"{self.domain}__{Path(self.instance_file).stem}__{self.social_law_label}"


@dataclass
class RunnerConfig:
    run_id: str
    output_dir: Path = DEFAULT_OUTPUT_DIR
    domains: List[str] = field(default_factory=list)
    instances: Dict[str, List[str]] = field(default_factory=dict)
    social_law_options: Tuple[bool, ...] = (False, True)
    verifiers: List[str] = field(default_factory=lambda: ["general"])
    limits: ResourceLimits = field(default_factory=ResourceLimits)


DOMAIN_SPECS: Dict[str, DomainSpec] = {
    "numeric_grid": DomainSpec(
        name="numeric_grid",
        generator_cls=NumericGridGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "numeric_grid",
        supports_social_law=True,
    ),
    "numeric_zenotravel": DomainSpec(
        name="numeric_zenotravel",
        generator_cls=NumericZenotravelGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "numeric_zenotravel",
        supports_social_law=True,
    ),
    "numeric_expedition": DomainSpec(
        name="numeric_expedition",
        generator_cls=ExpeditionGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "numeric_expedition",
        supports_social_law=True,
    ),
    "numeric_markettrader": DomainSpec(
        name="numeric_markettrader",
        generator_cls=MarketTraderGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "numeric_markettrader",
        supports_social_law=False,
    ),
    "numeric_civ": DomainSpec(
        name="numeric_civ",
        generator_cls=NumericCivGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "numeric_civ",
        supports_social_law=True,
    ),
    "grid": DomainSpec(
        name="grid",
        generator_cls=GridGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "grid",
        supports_social_law=True,
    ),
    "zenotravel": DomainSpec(
        name="zenotravel",
        generator_cls=ZenoTravelGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "zenotravel",
        supports_social_law=True,
    ),
    "driverlog": DomainSpec(
        name="driverlog",
        generator_cls=DriverLogGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "driverlog",
        supports_social_law=True,
    ),
    "blocksworld": DomainSpec(
        name="blocksworld",
        generator_cls=BlocksworldGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "blocksworld",
        supports_social_law=False,
    ),
    "intersection": DomainSpec(
        name="intersection",
        generator_cls=IntersectionProblemGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "intersection",
        supports_social_law=True,
    ),
    "numeric_intersection": DomainSpec(
        name="numeric_intersection",
        generator_cls=NumericIntersectionGenerator,
        instances_dir=PROJECT_ROOT / "experimentation" / "instances" / "intersection",
        supports_social_law=True,
    ),
}


VERIFIER_SPECS: Dict[str, VerifierSpec] = {
    "general": VerifierSpec(
        label="general",
        compiler_name="WaitingActionRobustnessVerifier",
        requires_snp=False,
    ),
    "simple": VerifierSpec(
        label="simple",
        compiler_name="SimpleNumericRobustnessVerifier",
        requires_snp=True,
    ),
}


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def _status_name(status) -> str:
    return status.name if hasattr(status, "name") else str(status)


def _sort_key(path: Path):
    stem = path.stem
    if stem.startswith("pfile") and stem[5:].isdigit():
        return (0, int(stem[5:]))
    return (1, stem)


def normalize_social_law_options(value) -> Tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ["with", "true"]:
            return (True,)
        if lowered in ["without", "false"]:
            return (False,)
        if lowered == "both":
            return (False, True)
    if isinstance(value, list):
        normalized = []
        for item in value:
            normalized.extend(list(normalize_social_law_options(item)))
        deduped = []
        for item in normalized:
            if item not in deduped:
                deduped.append(item)
        return tuple(deduped)
    raise ValueError(f"Unsupported social_law config: {value}")


def normalize_verifiers(value) -> List[str]:
    if isinstance(value, str):
        value = [value]
    normalized = []
    for item in value:
        lowered = item.lower()
        if lowered in ["general", "waiting", "waitingactionrobustnessverifier"]:
            normalized.append("general")
        elif lowered in ["simple", "simple_numeric", "simplenumericrobustnessverifier"]:
            normalized.append("simple")
        elif lowered == "both":
            normalized.extend(["general", "simple"])
        else:
            raise ValueError(f"Unsupported verifier config: {item}")
    deduped = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return deduped


def normalize_instance_values(values) -> List[str]:
    normalized = []
    for value in values:
        if isinstance(value, int):
            normalized.append(f"pfile{value}.json")
        else:
            file_name = str(value)
            if not file_name.endswith(".json"):
                file_name = f"{file_name}.json"
            normalized.append(file_name)
    return normalized


def discover_instance_files(domain: str, requested: Optional[List[str]]) -> List[str]:
    spec = DOMAIN_SPECS[domain]
    if requested is None:
        return [path.name for path in sorted(spec.instances_dir.glob("*.json"), key=_sort_key)]
    return normalize_instance_values(requested)


def parse_config(config_path: Path) -> RunnerConfig:
    data = json.loads(config_path.read_text())
    run_id = data.get("id", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(data.get("output_dir", DEFAULT_OUTPUT_DIR))

    problems_cfg = data.get("problems", {})
    domains = data.get("domains", problems_cfg.get("domains", sorted(DOMAIN_SPECS.keys())))
    limits_cfg = data.get("limits", data.get("resources", {}))

    if isinstance(domains, str):
        domains = [domains]

    instances_cfg = data.get("instances", problems_cfg.get("instances"))
    normalized_instances: Dict[str, List[str]] = {}
    if isinstance(instances_cfg, dict):
        for domain, values in instances_cfg.items():
            normalized_instances[domain] = discover_instance_files(domain, values)
    elif instances_cfg is not None:
        normalized = normalize_instance_values(instances_cfg)
        for domain in domains:
            normalized_instances[domain] = normalized

    config = RunnerConfig(
        run_id=run_id,
        output_dir=output_dir,
        domains=list(domains),
        instances=normalized_instances,
        social_law_options=normalize_social_law_options(
            data.get("social_law", problems_cfg.get("social_law", "both"))
        ),
        verifiers=normalize_verifiers(data.get("verifiers", data.get("compilations", ["general"]))),
        limits=ResourceLimits(
            engine=limits_cfg.get("engine", "enhsp"),
            planner_timeout_seconds=limits_cfg.get("planner_timeout_seconds", limits_cfg.get("planner_timeout")),
            wall_timeout_seconds=limits_cfg.get("wall_timeout_seconds", limits_cfg.get("timeout", 1800)),
            cpu_seconds=limits_cfg.get("cpu_seconds", limits_cfg.get("cpu_limit", 1800)),
            memory_bytes=limits_cfg.get("memory_bytes", limits_cfg.get("memory_limit", 16_000_000_000)),
        ),
    )
    return config


def build_cases(config: RunnerConfig) -> Tuple[List[ProblemCase], List[str]]:
    cases: List[ProblemCase] = []
    warnings: List[str] = []
    domain_to_instances: Dict[str, List[str]] = {}
    domain_specs: Dict[str, DomainSpec] = {}
    max_instances = 0

    for domain in config.domains:
        if domain not in DOMAIN_SPECS:
            raise ValueError(f"Unknown domain: {domain}")
        spec = DOMAIN_SPECS[domain]
        domain_specs[domain] = spec
        instance_files = config.instances.get(domain)
        if instance_files is None:
            instance_files = discover_instance_files(domain, None)
        domain_to_instances[domain] = instance_files
        max_instances = max(max_instances, len(instance_files))

    for instance_index in range(max_instances):
        for domain in config.domains:
            spec = domain_specs[domain]
            instance_files = domain_to_instances[domain]
            if instance_index >= len(instance_files):
                continue
            instance_file = instance_files[instance_index]
            for has_social_law in config.social_law_options:
                if has_social_law and not spec.supports_social_law:
                    warnings.append(
                        f"Skipping social law case for domain '{domain}' and instance '{instance_file}': domain has no social law."
                    )
                    continue
                cases.append(ProblemCase(domain=domain, instance_file=instance_file, has_social_law=has_social_law))
    return cases, warnings


def load_problem(case: ProblemCase):
    spec = DOMAIN_SPECS[case.domain]
    generator = spec.generator_cls()
    path = spec.instances_dir / case.instance_file
    return generator.generate_problem(str(path), sl=case.has_social_law)


def choose_engine(problem, default_engine: str) -> str:
    kind = problem.kind
    if kind.has_numeric_fluents() or kind.has_simple_numeric_planning() or kind.has_general_numeric_planning():
        return "enhsp"
    return "fast-downward"


def solve_problem(problem, engine_name: str, timeout_seconds: Optional[int]):
    stream = StringIO()
    selected_engine = choose_engine(problem, engine_name)
    with OneshotPlanner(name=selected_engine, problem_kind=problem.kind) as planner:
        result = planner.solve(problem, timeout=timeout_seconds, output_stream=stream)
    return result, stream.getvalue()


def classify_multi_agent_status(plan) -> SocialLawRobustnessStatus:
    status = SocialLawRobustnessStatus.ROBUST_RATIONAL
    for action_instance in plan.actions:
        prefix = action_instance.action.name.split("__")[0]
        if prefix and prefix[0] == "f":
            return SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
        if prefix and prefix[0] == "w":
            return SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK
    return status


def compile_for_verifier(problem, verifier: VerifierSpec):
    source_problem = problem
    conversion_log = StringIO()
    side_output = StringIO()
    with contextlib.redirect_stdout(side_output), contextlib.redirect_stderr(side_output):
        if verifier.requires_snp:
            conversion_log.write("=== SNP CONVERSION ===\n")
            source_problem = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
            conversion_log.write(f"Converted problem: {source_problem.name}\n\n")
        rbv = Compiler(
            name=verifier.compiler_name,
            problem_kind=source_problem.kind,
            compilation_kind=CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION,
        )
        rbv.skip_checks = True
        compiled = rbv.compile(source_problem).problem
    captured = side_output.getvalue().strip()
    if captured:
        conversion_log.write("=== COMPILER OUTPUT ===\n")
        conversion_log.write(captured)
        conversion_log.write("\n\n")
    return source_problem, compiled, conversion_log.getvalue()


def evaluate_problem(case: ProblemCase, verifier_label: str, limits: ResourceLimits):
    verifier = VERIFIER_SPECS[verifier_label]
    sections: List[Tuple[str, str]] = []
    started = time.time()
    warning_messages: List[str] = []

    problem = load_problem(case)
    sections.append(("CASE", "\n".join([
        f"domain: {case.domain}",
        f"instance_file: {case.instance_file}",
        f"has_social_law: {case.has_social_law}",
        f"problem_name: {problem.name}",
        f"verifier: {verifier.label}",
    ])))

    source_problem, compiled_problem, conversion_log = compile_for_verifier(problem, verifier)
    if conversion_log:
        sections.append(("CONVERSION", conversion_log))

    single_agent_statuses = []
    for agent in source_problem.agents:
        sap = SingleAgentProjection(agent)
        sap.skip_checks = True
        sa_problem = sap.compile(source_problem).problem
        result, log_text = solve_problem(sa_problem, limits.engine, limits.planner_timeout_seconds)
        status_name = _status_name(result.status)
        single_agent_statuses.append(f"{agent.name}:{status_name}")
        sections.append((f"SINGLE_AGENT {agent.name} {status_name}", log_text))
        if result.status not in POSITIVE_OUTCOMES:
            if result.status in [
                PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
                PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
            ]:
                final_status = SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT
            else:
                final_status = SocialLawRobustnessStatus.UNKNOWN
                warning_messages.append(
                    f"Single-agent projection for {agent.name} returned {status_name}."
                )
            elapsed = time.time() - started
            return {
                "case_id": case.case_id,
                "domain": case.domain,
                "instance": case.instance_file,
                "has_social_law": case.has_social_law,
                "verifier": verifier.label,
                "status": final_status.name,
                "single_agent_statuses": ";".join(single_agent_statuses),
                "compiled_problem_name": compiled_problem.name,
                "source_problem_name": source_problem.name,
                "planner_status": status_name,
                "elapsed_seconds": round(elapsed, 3),
                "warnings": " | ".join(warning_messages),
                "log_text": render_sections(sections),
            }

    compiled_result, compiled_log = solve_problem(compiled_problem, limits.engine, limits.planner_timeout_seconds)
    compiled_status_name = _status_name(compiled_result.status)
    sections.append((f"ROBUSTNESS {compiled_status_name}", compiled_log))

    if compiled_result.status in POSITIVE_OUTCOMES:
        final_status = classify_multi_agent_status(compiled_result.plan)
    elif compiled_result.status in [
        PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
        PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
    ]:
        final_status = SocialLawRobustnessStatus.ROBUST_RATIONAL
    else:
        final_status = SocialLawRobustnessStatus.UNKNOWN
        warning_messages.append(
            f"Compiled robustness problem returned {compiled_status_name}."
        )

    elapsed = time.time() - started
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "instance": case.instance_file,
        "has_social_law": case.has_social_law,
        "verifier": verifier.label,
        "status": final_status.name,
        "single_agent_statuses": ";".join(single_agent_statuses),
        "compiled_problem_name": compiled_problem.name,
        "source_problem_name": source_problem.name,
        "planner_status": compiled_status_name,
        "elapsed_seconds": round(elapsed, 3),
        "warnings": " | ".join(warning_messages),
        "log_text": render_sections(sections),
    }


def render_sections(sections: Iterable[Tuple[str, str]]) -> str:
    chunks = []
    for title, body in sections:
        chunks.append(f"=== {title} ===\n{body.rstrip()}\n")
    return "\n".join(chunks)


def _set_limits(limits: ResourceLimits):
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))


def _worker(case_data, verifier_label: str, limits_data, queue: Queue):
    try:
        limits = ResourceLimits(**limits_data)
        _set_limits(limits)
        case = ProblemCase(**case_data)
        result = evaluate_problem(case, verifier_label, limits)
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })


class RobustnessSuiteRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.run_dir = self.config.output_dir / self.config.run_id
        self.logs_dir = self.run_dir / "logs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv = self.run_dir / "results.csv"
        self.cases_csv = self.run_dir / "cases.csv"
        self.warnings_txt = self.run_dir / "warnings.txt"
        self.config_snapshot = self.run_dir / "config.snapshot.json"
        self.progress_log = self.run_dir / "progress.log"

    def write_config_snapshot(self):
        snapshot = {
            "run_id": self.config.run_id,
            "output_dir": str(self.config.output_dir),
            "domains": self.config.domains,
            "instances": self.config.instances,
            "social_law_options": list(self.config.social_law_options),
            "verifiers": self.config.verifiers,
            "limits": asdict(self.config.limits),
        }
        self.config_snapshot.write_text(json.dumps(snapshot, indent=2) + "\n")

    def reset_outputs(self):
        if self.results_csv.exists():
            self.results_csv.unlink()
        for path in self.logs_dir.glob("*.log"):
            path.unlink()

    def write_case_manifest(self, cases: List[ProblemCase], warnings: List[str]):
        with self.cases_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "domain", "instance", "has_social_law"])
            for case in cases:
                writer.writerow([case.case_id, case.domain, case.instance_file, case.has_social_law])
        self.warnings_txt.write_text("\n".join(warnings) + ("\n" if warnings else ""))

    def load_existing_results(self) -> List[Dict[str, object]]:
        if not self.results_csv.exists():
            return []
        with self.results_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if None in row:
                    continue
                row["elapsed_seconds"] = float(row["elapsed_seconds"]) if row["elapsed_seconds"] else 0
                rows.append(row)
            return rows

    def append_result(self, row: Dict[str, object]):
        file_exists = self.results_csv.exists()
        with self.results_csv.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "domain",
                    "instance",
                    "has_social_law",
                    "verifier",
                    "status",
                    "planner_status",
                    "single_agent_statuses",
                    "elapsed_seconds",
                    "source_problem_name",
                    "compiled_problem_name",
                    "warnings",
                    "log_file",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_path_for(self, case: ProblemCase, verifier_label: str) -> Path:
        return self.logs_dir / f"{_safe_name(verifier_label)}__{_safe_name(case.case_id)}.log"

    def run_case(self, case: ProblemCase, verifier_label: str) -> Dict[str, object]:
        queue: Queue = Queue()
        process = Process(
            target=_worker,
            args=(asdict(case), verifier_label, asdict(self.config.limits), queue),
        )
        process.start()
        process.join(self.config.limits.wall_timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "case_id": case.case_id,
                "domain": case.domain,
                "instance": case.instance_file,
                "has_social_law": case.has_social_law,
                "verifier": verifier_label,
                "status": SocialLawRobustnessStatus.UNKNOWN.name,
                "planner_status": "WALL_TIMEOUT",
                "single_agent_statuses": "",
                "elapsed_seconds": self.config.limits.wall_timeout_seconds,
                "source_problem_name": "",
                "compiled_problem_name": "",
                "warnings": "Worker hit wall timeout.",
                "log_text": "Worker terminated after wall timeout.\n",
            }

        if queue.empty():
            return {
                "case_id": case.case_id,
                "domain": case.domain,
                "instance": case.instance_file,
                "has_social_law": case.has_social_law,
                "verifier": verifier_label,
                "status": SocialLawRobustnessStatus.UNKNOWN.name,
                "planner_status": "NO_RESULT",
                "single_agent_statuses": "",
                "elapsed_seconds": 0,
                "source_problem_name": "",
                "compiled_problem_name": "",
                "warnings": "Worker exited without producing a result.",
                "log_text": "Worker exited without producing a result.\n",
            }

        worker_result = queue.get()
        if worker_result["ok"]:
            return worker_result["result"]
        return {
            "case_id": case.case_id,
            "domain": case.domain,
            "instance": case.instance_file,
            "has_social_law": case.has_social_law,
            "verifier": verifier_label,
            "status": SocialLawRobustnessStatus.UNKNOWN.name,
            "planner_status": "WORKER_ERROR",
            "single_agent_statuses": "",
            "elapsed_seconds": 0,
            "source_problem_name": "",
            "compiled_problem_name": "",
            "warnings": worker_result["error"],
            "log_text": worker_result["error"] + "\n",
        }

    def run(self, resume: bool = False) -> List[Dict[str, object]]:
        cases, warnings = build_cases(self.config)
        existing_results: List[Dict[str, object]] = []
        completed_pairs = set()
        if resume:
            existing_results = self.load_existing_results()
            completed_pairs = {
                (result["case_id"], result["verifier"])
                for result in existing_results
            }
            if not self.cases_csv.exists() or not self.warnings_txt.exists():
                self.write_case_manifest(cases, warnings)
            if not self.config_snapshot.exists():
                self.write_config_snapshot()
        else:
            self.reset_outputs()
            self.write_config_snapshot()
            self.write_case_manifest(cases, warnings)
            if self.progress_log.exists():
                self.progress_log.unlink()

        total_pairs = len(cases) * len(self.config.verifiers)
        reporter = ProgressReporter(self.progress_log)
        if resume and existing_results:
            reporter.emit(
                f"Resuming robustness suite '{self.config.run_id}' with {len(existing_results)}/{total_pairs} "
                f"completed case/verifier pairs already on disk."
            )
        else:
            reporter.emit(
                f"Starting robustness suite '{self.config.run_id}' with {len(cases)} cases and "
                f"{len(self.config.verifiers)} verifiers ({total_pairs} case/verifier pairs)."
            )
        reporter.emit(f"Run directory: {self.run_dir}")
        reporter.emit(f"Results CSV: {self.results_csv}")
        reporter.emit(f"Case manifest: {self.cases_csv}")
        reporter.emit(f"Warnings file: {self.warnings_txt}")
        reporter.emit(
            "Limits: "
            f"engine={self.config.limits.engine}, "
            f"planner_timeout_seconds={self.config.limits.planner_timeout_seconds}, "
            f"wall_timeout_seconds={self.config.limits.wall_timeout_seconds}, "
            f"cpu_seconds={self.config.limits.cpu_seconds}, "
            f"memory_bytes={self.config.limits.memory_bytes}"
        )
        if warnings:
            reporter.emit(f"Case generation warnings: {len(warnings)}")

        results = list(existing_results)
        completed = len(completed_pairs)
        status_counts: Dict[str, int] = {}
        for result in existing_results:
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        try:
            for case_index, case in enumerate(cases, start=1):
                remaining_verifiers = [
                    verifier_label
                    for verifier_label in self.config.verifiers
                    if (case.case_id, verifier_label) not in completed_pairs
                ]
                if not remaining_verifiers:
                    continue
                reporter.emit(
                    f"Case {case_index}/{len(cases)}: domain={case.domain}, instance={case.instance_file}, "
                    f"social_law={case.social_law_label}"
                )
                for verifier_index, verifier_label in enumerate(self.config.verifiers, start=1):
                    if (case.case_id, verifier_label) in completed_pairs:
                        continue
                    pair_index = completed + 1
                    reporter.emit(
                        f"Starting pair {pair_index}/{total_pairs}: case_id={case.case_id}, "
                        f"verifier={verifier_label} ({verifier_index}/{len(self.config.verifiers)} for this case)"
                    )
                    started = time.time()
                    result = self.run_case(case, verifier_label)
                    elapsed = time.time() - started
                    log_path = self.log_path_for(case, verifier_label)
                    log_path.write_text(result.pop("log_text"))
                    result["log_file"] = str(log_path)
                    self.append_result(result)
                    results.append(result)
                    completed += 1
                    completed_pairs.add((case.case_id, verifier_label))
                    status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
                    reporter.emit(
                        f"Finished pair {completed}/{total_pairs}: case_id={case.case_id}, "
                        f"verifier={verifier_label}, status={result['status']}, "
                        f"planner_status={result['planner_status']}, elapsed_seconds={result['elapsed_seconds']}, "
                        f"wall_clock_seconds={round(elapsed, 3)}, log_file={log_path}"
                    )
                    if result["warnings"]:
                        reporter.emit(
                            f"Warnings for {case.case_id}/{verifier_label}: {result['warnings']}"
                        )
        finally:
            reporter.emit(f"Run stopped after {completed}/{total_pairs} pairs.")
            if status_counts:
                summary = ", ".join(
                    f"{status}={count}" for status, count in sorted(status_counts.items())
                )
                reporter.emit(f"Status summary so far: {summary}")
            reporter.close()
        return results


def build_runner_from_config_path(config_path: Path) -> RobustnessSuiteRunner:
    return RobustnessSuiteRunner(parse_config(config_path))


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Run robustness experiments from a JSON config.")
    parser.add_argument("config", help="Path to JSON config file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory by keeping prior results and continuing unfinished case/verifier pairs.",
    )
    return parser


def main(argv=None):
    parser = make_arg_parser()
    args = parser.parse_args(argv)
    runner = build_runner_from_config_path(Path(args.config).resolve())
    results = runner.run(resume=args.resume)
    print(f"Run directory: {runner.run_dir}")
    print(f"Cases: {len(results)}")
    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main(sys.argv[1:])

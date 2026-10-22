"""Run the backend configuration and integration contracts on any hosted OS."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
PROFILES = (
    ("default", False, ()),
    ("native-cpu", True, ("native-cpu",)),
    ("dual-cpu", False, ("native-cpu",)),
    ("candle-cpu-native-family", False, ("native",)),
    ("candle-family", True, ("candle",)),
    ("native-family", True, ("native",)),
    ("both-families", True, ("candle", "native")),
    ("internal-cpu", True, ("candle", "native-cpu")),
)


class ContractError(RuntimeError):
    """An observed command result violates a backend contract."""


def check_rejection(returncode, stdout, backend_package_id):
    if returncode != 101:
        raise ContractError(f"expected configuration rejection (101), got {returncode}")
    try:
        messages = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ContractError("invalid Cargo JSON output") from error
    if not messages or any(not isinstance(message, dict) for message in messages):
        raise ContractError("missing Cargo JSON messages")
    diagnostics = [message for message in messages if message.get("reason") == "compiler-message"]
    levels = {"error", "warning", "note", "help", "failure-note", "ice"}
    for entry in diagnostics:
        diagnostic = entry.get("message")
        if (
            not isinstance(entry.get("package_id"), str)
            or not isinstance(diagnostic, dict)
            or not isinstance(diagnostic.get("message"), str)
            or not isinstance(diagnostic.get("level"), str)
            or diagnostic["level"] not in levels
            or "code" not in diagnostic
        ):
            raise ContractError("malformed compiler diagnostic")
        code = diagnostic["code"]
        if code is not None and (not isinstance(code, dict) or not isinstance(code.get("code"), str)):
            raise ContractError("malformed compiler diagnostic code")
    errors = [message for message in diagnostics if message["message"]["level"] in {"error", "ice"}]
    finished = [message for message in messages if message.get("reason") == "build-finished"]
    if len(finished) != 1 or finished[0].get("success") is not False:
        raise ContractError("missing failed build-finished record")
    if len(errors) != 1:
        raise ContractError(f"expected exactly one configuration error, got {len(errors)}")
    error = errors[0]
    if (
        error.get("package_id") != backend_package_id
        or error["message"]["level"] != "error"
        or error["message"].get("code") is not None
        or "enable candle or native" not in error["message"].get("message", "")
    ):
        raise ContractError("unexpected compiler error instead of the backend configuration error")


def check_native_tree(output, root):
    packages = [line.split()[0] for line in output.splitlines() if line.strip()]
    if not packages or packages[0] != root:
        raise ContractError(f"missing dependency tree rooted at {root}")
    forbidden = {"nove_candle", "candle-core"}.intersection(packages)
    if forbidden:
        raise ContractError(f"Native-only normal dependencies contain {sorted(forbidden)}")


def check_tensor_metadata(metadata):
    packages = [package for package in metadata.get("packages", []) if package.get("name") == "nove_tensor"]
    if len(packages) != 1 or not isinstance(packages[0].get("dependencies"), list):
        raise ContractError("missing nove_tensor package metadata")
    if any(dependency.get("name") == "candle-core" for dependency in packages[0]["dependencies"]):
        raise ContractError("nove_tensor directly depends on candle-core")


def feature_args(profile, packages):
    _, no_default, features = profile
    args = ["--no-default-features"] if no_default else []
    if features:
        selected = features if len(packages) == 1 else tuple(
            f"{package}/{feature}" for package in packages for feature in features
        )
        args += ["--features", ",".join(selected)]
    return args


class Runner:
    def __init__(self, group, offline):
        self.group = group
        self.offline = offline
        output = ROOT / "target" / "backend-contract-checks"
        output.mkdir(parents=True, exist_ok=True)
        self.output = Path(tempfile.mkdtemp(prefix=group + "-", dir=output))
        self.env = os.environ.copy()
        self.env["CARGO_TERM_COLOR"] = "never"
        if os.name == "nt":
            tmp = self.output / "tmp"
            tmp.mkdir()
            self.env["TEMP"] = self.env["TMP"] = str(tmp)
        self.records = []

    def save(self, status, error=None):
        summary = {"group": self.group, "status": status, "error": error, "commands": self.records}
        (self.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def cargo(self, label, args, *, expected=0, env=None):
        command = ["cargo", args[0], "--locked"]
        if self.offline:
            command.append("--offline")
        command += args[1:]
        print(f"\n[{label}] {shlex.join(command)}", flush=True)
        started = time.monotonic()
        result = subprocess.run(
            command, cwd=ROOT, env=self.env | (env or {}), capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        log = self.output / f"{len(self.records) + 1:02}-{label}.log"
        log.write_text("STDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        print(f"[{label}] exit={result.returncode}; log={log}", flush=True)
        self.records.append({
            "label": label, "command": command, "exit": result.returncode,
            "expected_exit": expected, "seconds": round(time.monotonic() - started, 2),
            "log": str(log), "outcome": "command_succeeded" if result.returncode == 0 else "not_validated",
        })
        self.save("running")
        if result.returncode != expected:
            raise ContractError(f"{label}: expected exit {expected}, got {result.returncode}")
        return result

    def libraries(self):
        for profile in PROFILES:
            for package in ("nove_backend", "nove_tensor"):
                self.cargo(f"library-{package}-{profile[0]}", [
                    "check", "-p", package, "--lib", *feature_args(profile, (package,)),
                ])

    def cpu(self):
        packages = ("nove_backend", "nove_tensor")
        for profile in PROFILES[:4]:
            flags = feature_args(profile, packages)
            selected = ["-p", "nove_backend", "-p", "nove_tensor"]
            self.cargo(f"cpu-{profile[0]}-tests", ["test", *selected, *flags, "--lib", "--tests"])
            self.cargo(f"cpu-{profile[0]}-docs", ["test", *selected, *flags, "--doc"])
            self.cargo(f"cpu-{profile[0]}-clippy", [
                "clippy", *selected, *flags, "--all-targets", "--", "-D", "warnings",
            ])
        flags = self.env.get("RUSTDOCFLAGS", "") + " -D rustdoc::broken_intra_doc_links"
        self.cargo("backend-doc-links", [
            "doc", "-p", "nove_backend", "--features", "native-cpu", "--no-deps",
        ], env={"RUSTDOCFLAGS": flags.strip()})

    def families(self):
        for profile in PROFILES[4:]:
            self.cargo(f"family-{profile[0]}", [
                "test", "-p", "nove_backend", *feature_args(profile, ("nove_backend",)),
                "--test", "device_contract_tests", "--test", "serialization_contract_tests",
            ])

    def isolation(self):
        for no_default in (False, True):
            flags = ["--no-default-features"] if no_default else []
            name = "no-default" if no_default else "default"
            self.cargo(f"external-cpu-backend-{name}", [
                "test", "-p", "nove_backend", "-p", "nove_native", *flags,
                "--features", "nove_backend/native,nove_native/cpu", "--lib", "--tests",
            ])
            self.cargo(f"external-cpu-tensor-{name}", [
                "check", "-p", "nove_tensor", "-p", "nove_native", *flags,
                "--features", "nove_tensor/native,nove_native/cpu", "--lib",
            ])
        result = self.cargo("dependency-metadata", ["metadata", "--format-version", "1", "--no-deps"])
        metadata = json.loads(result.stdout)
        check_tensor_metadata(metadata)
        backend_id = next(package["id"] for package in metadata["packages"] if package["name"] == "nove_backend")
        for package in ("nove_backend", "nove_tensor"):
            result = self.cargo(f"no-family-{package}", [
                "check", "-p", package, "--no-default-features", "--lib", "--message-format=json",
            ], expected=101)
            check_rejection(result.returncode, result.stdout, backend_id)
            self.records[-1]["outcome"] = "expected_configuration_rejection"
            self.save("running")
            print(f"{package}: expected configuration rejection verified (not a successful build)", flush=True)
            result = self.cargo(f"native-tree-{package}", [
                "tree", "-p", package, "--no-default-features", "--features", "native-cpu",
                "--edges", "normal", "--prefix", "none", "--format", "{p}",
            ])
            check_native_tree(result.stdout, package)

    def model(self):
        self.cargo("model-load-contracts", [
            "test", "-p", "nove_model", "--features", "native-cpu", "--test", "load_contract_tests",
        ])


def main():
    groups = ("libraries", "cpu", "families", "isolation", "model")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=(*groups, "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    runner = Runner(args.group, args.offline)
    print(f"Backend contract results: {runner.output}", flush=True)
    try:
        for group in groups if args.group == "all" else (args.group,):
            getattr(runner, group)()
    except (ContractError, OSError, ValueError, KeyError, StopIteration) as error:
        runner.save("failed", str(error))
        print(f"Backend contracts FAILED: {error}", file=sys.stderr)
        return 1
    runner.save("passed")
    print(f"Backend contracts passed; results: {runner.output / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

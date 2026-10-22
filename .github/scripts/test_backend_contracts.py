"""Regression tests for CI guards that must not report false success."""

import copy
import importlib
import importlib.util
import json
import unittest


BACKEND_ID = "path+file:///workspace/crates/nove_backend#0.1.2"


def compiler_error(package_id=BACKEND_ID, message="enable candle or native", code=None):
    return {
        "reason": "compiler-message",
        "package_id": package_id,
        "manifest_path": "/workspace/crates/nove_backend/Cargo.toml",
        "target": {
            "kind": ["lib"], "crate_types": ["lib"], "name": "nove_backend",
            "src_path": "/workspace/crates/nove_backend/src/lib.rs",
            "edition": "2024", "doc": True, "doctest": True, "test": True,
        },
        "message": {
            "$message_type": "diagnostic", "level": "error", "message": message,
            "code": code, "spans": [], "children": [], "rendered": "error: " + message,
        },
    }


def cargo_output(*messages):
    return "\n".join(json.dumps(message) for message in messages)


class ContractGuards(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("check_backend_contracts"),
            "the backend contract checker has not been implemented",
        )
        self.checks = importlib.import_module("check_backend_contracts")
        self.rejection = cargo_output(
            compiler_error(), {"reason": "build-finished", "success": False}
        )

    def test_accepts_only_the_expected_configuration_rejection(self):
        self.checks.check_rejection(101, self.rejection, BACKEND_ID)

    def test_rejects_unexpected_success(self):
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(0, self.rejection, BACKEND_ID)

    def test_rejects_non_compilation_exit(self):
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(1, self.rejection, BACKEND_ID)

    def test_rejects_unrelated_compiler_error(self):
        output = cargo_output(
            compiler_error(message="cannot resolve module", code={"code": "E0433"}),
            {"reason": "build-finished", "success": False},
        )
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_correct_text_from_wrong_package(self):
        output = cargo_output(
            compiler_error(package_id="registry+example#another_crate@1.0.0"),
            {"reason": "build-finished", "success": False},
        )
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_extra_compiler_errors(self):
        output = cargo_output(
            compiler_error(), compiler_error(message="non-exhaustive match"),
            {"reason": "build-finished", "success": False},
        )
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_missing_error_json(self):
        for output in ["", cargo_output({"reason": "build-finished", "success": False})]:
            with self.subTest(output=output), self.assertRaises(self.checks.ContractError):
                self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_malformed_json(self):
        for output in ["{broken", self.rejection + "\nnot json", "null"]:
            with self.subTest(output=output), self.assertRaises(self.checks.ContractError):
                self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_non_text_diagnostic_message(self):
        malformed = compiler_error()
        malformed["message"]["message"] = ["enable candle or native"]
        output = cargo_output(malformed, {"reason": "build-finished", "success": False})
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_incomplete_additional_diagnostic(self):
        output = cargo_output(
            compiler_error(), {"reason": "compiler-message", "message": {}},
            {"reason": "build-finished", "success": False},
        )
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_unknown_additional_diagnostic_level(self):
        unexpected = compiler_error(message="unexpected diagnostic")
        unexpected["message"]["level"] = "unknown"
        output = cargo_output(
            compiler_error(), unexpected, {"reason": "build-finished", "success": False}
        )
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_rejection(101, output, BACKEND_ID)

    def test_rejects_incomplete_or_successful_build_record(self):
        for end in [[], [{"reason": "build-finished", "success": True}]]:
            with self.subTest(end=end), self.assertRaises(self.checks.ContractError):
                self.checks.check_rejection(101, cargo_output(compiler_error(), *end), BACKEND_ID)

    def test_warning_does_not_count_as_an_additional_error(self):
        warning = compiler_error(message="unused variable")
        warning["message"]["level"] = "warning"
        output = cargo_output(
            warning, compiler_error(), {"reason": "build-finished", "success": False}
        )
        self.checks.check_rejection(101, output, BACKEND_ID)

    def test_accepts_native_dependency_tree(self):
        self.checks.check_native_tree(
            "nove_tensor v0.1.2 (/workspace/crates/nove_tensor)\n"
            "nove_backend v0.1.2 (/workspace/crates/nove_backend)\n"
            "nove_native v0.1.2 (/workspace/crates/nove_native)\n"
            "nove_cpu v0.1.2 (/workspace/crates/nove_cpu)\n",
            "nove_tensor",
        )

    def test_rejects_candle_in_native_dependency_tree(self):
        for package in ["nove_candle", "candle-core"]:
            output = f"nove_backend v0.1.2 (/workspace)\n{package} v0.9.1\n"
            with self.subTest(package=package), self.assertRaises(self.checks.ContractError):
                self.checks.check_native_tree(output, "nove_backend")

    def test_rejects_empty_or_wrong_root_dependency_tree(self):
        for output in ["", "another_crate v1.0.0\n"]:
            with self.subTest(output=output), self.assertRaises(self.checks.ContractError):
                self.checks.check_native_tree(output, "nove_tensor")

    def test_allows_transitive_candle_dependency(self):
        self.checks.check_tensor_metadata({"packages": [
            {"name": "nove_tensor", "dependencies": [{"name": "nove_backend", "kind": None}]},
            {"name": "nove_candle", "dependencies": [{"name": "candle-core", "kind": None}]},
        ]})

    def test_rejects_direct_candle_dependency_even_if_optional(self):
        metadata = {"packages": [{"name": "nove_tensor", "dependencies": []}]}
        for kind in [None, "dev", "build"]:
            changed = copy.deepcopy(metadata)
            changed["packages"][0]["dependencies"].append(
                {"name": "candle-core", "kind": kind, "optional": True}
            )
            with self.subTest(kind=kind), self.assertRaises(self.checks.ContractError):
                self.checks.check_tensor_metadata(changed)

    def test_rejects_metadata_without_tensor(self):
        with self.assertRaises(self.checks.ContractError):
            self.checks.check_tensor_metadata({"packages": []})


if __name__ == "__main__":
    unittest.main()

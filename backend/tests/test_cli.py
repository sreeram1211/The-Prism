"""
CLI integration tests using Typer's CliRunner.

Coverage:
  - prism --version
  - prism resolve <model_id>  (mocked HF download)
  - prism resolve --json
  - prism scan <model_id>     (mocked HF + mock engine)
  - prism scan --json
  - prism scan --dim (dimension filter)
  - prism scan (unknown dimension → exit 1)
  - prism info
  - prism generate / monitor / agent  (Phase stubs → exit 0)
  - prism resolve (model not found → exit 1)
  - prism resolve (gated model → exit 1)
  - prism resolve (invalid model_id → validation error)
"""

from __future__ import annotations

import json
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from prism.cli.main import app
from prism.resolver.arch_detector import ArchitectureFamily

from .conftest import LLAMA3_8B_CONFIG, MISTRAL_7B_CONFIG, MAMBA_130M_CONFIG

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp_config(config: dict[str, Any]) -> str:
    """Write a config dict to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return f.name


def _mock_hf(config: dict[str, Any]):
    """Context-manager-compatible patch for hf_hub_download."""
    tmpfile = _write_tmp_config(config)
    return patch("prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile)


# ---------------------------------------------------------------------------
# prism --version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Prism" in result.output
        assert "0.6.0" in result.output

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "0.6.0" in result.output


# ---------------------------------------------------------------------------
# prism resolve
# ---------------------------------------------------------------------------

class TestResolveCommand:
    def test_resolve_success_llama(self):
        with _mock_hf(LLAMA3_8B_CONFIG):
            result = runner.invoke(app, ["resolve", "meta-llama/Meta-Llama-3-8B"])
        assert result.exit_code == 0
        assert "llama" in result.output.lower()

    def test_resolve_success_mamba(self):
        with _mock_hf(MAMBA_130M_CONFIG):
            result = runner.invoke(app, ["resolve", "state-spaces/mamba-130m-hf"])
        assert result.exit_code == 0
        # SSM family should appear
        assert "ssm" in result.output.lower() or "SSM" in result.output

    def test_resolve_json_output(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(app, ["resolve", "--json", "mistralai/Mistral-7B-v0.1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model_type"] == "mistral"
        assert data["family"] == "attention"
        assert isinstance(data["lora_targets"], list)
        assert "q_proj" in data["lora_targets"]

    def test_resolve_json_contains_all_keys(self):
        with _mock_hf(LLAMA3_8B_CONFIG):
            result = runner.invoke(app, ["resolve", "--json", "org/model"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for key in [
            "model_id", "revision", "model_type", "family",
            "num_hidden_layers", "hidden_size", "lora_targets",
            "lora_targets_minimal", "lora_rank_recommendation",
            "param_count_estimate", "model_size_gb_bf16",
        ]:
            assert key in data, f"Missing key: {key!r}"

    def test_resolve_with_revision(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(app, ["resolve", "--revision", "v0.1", "mistralai/Mistral-7B-v0.1"])
        assert result.exit_code == 0

    def test_resolve_model_not_found_exits_1(self):
        from huggingface_hub.utils import RepositoryNotFoundError as HFNotFound
        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFNotFound(message="not found", response=MagicMock()),
        ):
            result = runner.invoke(app, ["resolve", "no-one/does-not-exist"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_resolve_gated_model_exits_1(self):
        from huggingface_hub.utils import GatedRepoError as HFGated
        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFGated(message="gated", response=MagicMock()),
        ):
            result = runner.invoke(app, ["resolve", "meta-llama/Llama-2-7b-hf"])
        assert result.exit_code == 1
        assert "gated" in result.output.lower()

    def test_resolve_missing_model_id(self):
        result = runner.invoke(app, ["resolve"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# prism scan
# ---------------------------------------------------------------------------

class TestScanCommand:
    def test_scan_success(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(app, ["scan", "mistralai/Mistral-7B-v0.1"])
        assert result.exit_code == 0
        # Dimension names should appear in the output
        for dim in ["Sycophancy", "Hedging", "Calibration", "Depth", "Coherence"]:
            assert dim in result.output, f"Missing dimension: {dim}"

    def test_scan_json_output(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(app, ["scan", "--json", "mistralai/Mistral-7B-v0.1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "scores" in data
        assert len(data["scores"]) == 9
        assert "geometric_separation_ratio" in data
        assert 125 <= data["geometric_separation_ratio"] <= 1376

    def test_scan_json_scores_in_range(self):
        with _mock_hf(LLAMA3_8B_CONFIG):
            result = runner.invoke(app, ["scan", "--json", "meta-llama/Meta-Llama-3-8B"])
        data = json.loads(result.output)
        for s in data["scores"]:
            assert 0.0 <= s["score"] <= 1.0, f"{s['dimension']} score out of range"

    def test_scan_dimension_filter(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(
                app,
                ["scan", "--json", "--dim", "sycophancy", "--dim", "depth",
                 "mistralai/Mistral-7B-v0.1"],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["scores"]) == 2
        dims = {s["dimension"] for s in data["scores"]}
        assert dims == {"sycophancy", "depth"}

    def test_scan_unknown_dimension_exits_1(self):
        with _mock_hf(MISTRAL_7B_CONFIG):
            result = runner.invoke(
                app,
                ["scan", "--dim", "not_a_real_dim", "mistralai/Mistral-7B-v0.1"],
            )
        assert result.exit_code == 1
        assert "unknown dimension" in result.output.lower()

    def test_scan_deterministic_across_invocations(self):
        """Two scan runs on the same model must produce identical JSON."""
        with _mock_hf(LLAMA3_8B_CONFIG):
            r1 = runner.invoke(app, ["scan", "--json", "meta-llama/Meta-Llama-3-8B"])
        with _mock_hf(LLAMA3_8B_CONFIG):
            r2 = runner.invoke(app, ["scan", "--json", "meta-llama/Meta-Llama-3-8B"])
        d1 = json.loads(r1.output)
        d2 = json.loads(r2.output)
        for s1, s2 in zip(d1["scores"], d2["scores"]):
            assert s1["score"] == s2["score"]

    def test_scan_mamba_uses_ssm_family(self):
        """Mamba models should be resolved as SSM and bias scores accordingly."""
        with _mock_hf(MAMBA_130M_CONFIG):
            result = runner.invoke(app, ["scan", "--json", "state-spaces/mamba-130m-hf"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Just verify it ran without error and returned 9 scores
        assert len(data["scores"]) == 9

    def test_scan_missing_model_id(self):
        result = runner.invoke(app, ["scan"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# prism info
# ---------------------------------------------------------------------------

class TestInfoCommand:
    def test_info_runs_without_error(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0

    def test_info_shows_version(self):
        result = runner.invoke(app, ["info"])
        assert "0.6.0" in result.output

    def test_info_shows_feature_table(self):
        result = runner.invoke(app, ["info"])
        for feature in ["Prism Resolve", "Prism Scan", "Prism Generate"]:
            assert feature in result.output

    def test_info_with_license_key(self):
        result = runner.invoke(app, ["info", "--license-key", "PRISM-PRO-TESTKEY1"])
        assert result.exit_code == 0
        # Pro license key should show PRO tier
        assert "PRO" in result.output


# ---------------------------------------------------------------------------
# Phase stub commands
# ---------------------------------------------------------------------------

class TestPhaseStubs:
    def test_generate_exits_0(self):
        result = runner.invoke(app, ["generate", "org/model"])
        assert result.exit_code == 0
        assert "Phase 3" in result.output

    def test_monitor_exits_0(self):
        result = runner.invoke(app, ["monitor", "org/model"])
        assert result.exit_code == 0
        assert "Phase 4" in result.output

    def test_agent_exits_0(self):
        result = runner.invoke(app, ["agent"])
        assert result.exit_code == 0
        assert "Phase 5" in result.output

    def test_generate_mentions_pro(self):
        result = runner.invoke(app, ["generate", "org/model"])
        assert "Pro" in result.output

    def test_monitor_mentions_pro(self):
        result = runner.invoke(app, ["monitor", "org/model"])
        assert "Pro" in result.output

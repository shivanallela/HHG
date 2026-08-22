import subprocess
import sys
from pathlib import Path

import pytest

@pytest.mark.parametrize("strategy", ["sentence", "word"])
def test_benchmark_cli_markdown(tmp_path: Path, strategy: str):
    """Run the benchmark CLI with a single query and verify JSON and markdown reports are created."""
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    # Build command arguments
    cmd = [
        sys.executable,
        "-m",
        "evaluation.benchmark_retrieval",
        "--num-queries",
        "1",
        f"--output={output_json}",
        f"--output-md={output_md}",
        f"--strategy={strategy}",
    ]
    # Execute the CLI
    result = subprocess.run(cmd, cwd="c:/HHG", capture_output=True, text=True)
    # Ensure the process exited successfully
    assert result.returncode == 0, f"CLI failed with exit code {result.returncode}: {result.stderr}"
    # Verify output files exist
    assert output_json.is_file(), "JSON report file was not created"
    assert output_md.is_file(), "Markdown report file was not created"
    # Basic sanity check on markdown content
    md_content = output_md.read_text(encoding="utf-8")
    assert "# Retrieval Latency Benchmark Report" in md_content
    assert "Strategy" in md_content

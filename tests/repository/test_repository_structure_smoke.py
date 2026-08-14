import csv
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

OLD_PATH_PATTERN = re.compile(
    r"Final[\\/]|Hardware Deployment Run|Edge-IIOT-run|Codistillation[\\/]|"
    r"old-routes|Repository_Archive|"
    r"deployment[\\/]hardware_hil[\\/](results|reports|compile_logs)|"
    r"(?<!results[\\/])hardware_hil[\\/](results|reports|compile_logs)|"
    r"(?<!deployment[\\/]firmware_export[\\/]wsnds_rfkd_hil[\\/])"
    r"(?<!deployment[\\/]msp430[\\/])hardware_export[\\/]"
)

TEXT_SUFFIXES = {".py", ".md", ".c", ".h"}
EXPECTED_TOP_LEVEL = {
    ".gitattributes",
    ".gitignore",
    "ARTIFACT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "data",
    "deployment",
    "docs",
    "experiments",
    "LICENSE",
    "manuscript",
    "NOTICE.md",
    "pytest.ini",
    "README.md",
    "requirements.txt",
    "research_history",
    "results",
    "tests",
}


def is_excluded_from_active_scan(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    rel = path.as_posix()
    if rel == "tests/repository/test_repository_structure_smoke.py":
        return True
    if rel.startswith("results/wsnds/leakage_free_rerun/_codex_session_"):
        return True
    if path.suffix not in TEXT_SUFFIXES:
        return True
    if "research_history" in parts:
        return True
    if "build" in parts:
        return True
    if any(part.startswith("generated_") for part in path.parts):
        return True
    if rel.startswith("docs/repository/"):
        return True
    return False


class RepositoryStructureSmokeTests(unittest.TestCase):
    def test_tracked_top_level_layout_is_professional(self):
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        top_levels = {
            raw.decode("utf-8").split("/", 1)[0]
            for raw in proc.stdout.split(b"\0")
            if raw
        }
        self.assertEqual(top_levels, EXPECTED_TOP_LEVEL)

    def test_active_files_do_not_reference_old_layout_paths(self):
        offenders = []
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        tracked_paths = [
            Path(raw.decode("utf-8"))
            for raw in proc.stdout.split(b"\0")
            if raw
        ]
        for rel_path in tracked_paths:
            if is_excluded_from_active_scan(rel_path):
                continue
            path = ROOT / rel_path
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if OLD_PATH_PATTERN.search(text):
                offenders.append(rel_path.as_posix())
        self.assertEqual(offenders, [])

    def test_path_reference_audit_has_no_active_review_rows(self):
        audit = (
            ROOT
            / "research_history"
            / "documentation_snapshots"
            / "repository_restructure"
            / "path_reference_audit.csv"
        )
        with audit.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        active_rows = [row for row in rows if row["category"] == "active_review"]
        self.assertEqual(active_rows, [])

    def test_research_history_is_explicitly_historical(self):
        readme = ROOT / "research_history" / "README.md"
        text = readme.read_text(encoding="utf-8").lower()
        for token in ["historical", "traceability", "not the current runnable entrypoints"]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()

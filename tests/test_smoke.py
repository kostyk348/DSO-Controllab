from dso_controllab.experiment import run_suite


def test_run_suite_smoke() -> None:
    report = run_suite(worlds=3, seed=1, steps=80)
    assert "DSO" in report["summary"]
    assert report["summary"]["DSO"]["contract_pass_rate"] == 1.0
    assert report["summary"]["DSO"]["resource_pass_rate"] == 1.0
    assert len(report["rows"]) == 12
    assert all(row["verification_passed"] == 1.0 for row in report["rows"] if row["controller"] == "DSO")

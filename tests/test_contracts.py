"""Контракты DSO-ControlLab: проверки датаклассов без numpy и без pytest.

Эти тесты запускаются голым стандартным Python:

    python -m unittest discover -s tests -p "test_contracts.py" -v
    # или просто: python tests/test_contracts.py

Зачем отдельно от test_smoke.py: смоук-тест тянет numpy через experiment/world и падает
на машине без зависимостей. Контракты — чистая логика, и их стоит проверять всегда:
именно на них опирается вердикт «план прошёл/не прошёл».
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dso_controllab.contracts import (  # noqa: E402
    ControlContract,
    DeploymentContract,
    ResourceContract,
    RuntimeContract,
    VerificationResult,
)


class TestDefaults(unittest.TestCase):
    def test_resource_defaults_are_sane(self) -> None:
        rc = ResourceContract()
        self.assertGreater(rc.cycles_max, 0)
        self.assertGreater(rc.ram_bytes_max, 0)
        self.assertGreater(rc.wcet_us_max, 0.0)
        self.assertGreaterEqual(rc.jitter_us_max, 0.0)
        # WCET не может быть больше периода цикла в микросекундах — иначе план заведомо невыполним
        self.assertLess(rc.wcet_us_max, rc.cycles_max * 1000.0)

    def test_deployment_bundles_all_three(self) -> None:
        dc = DeploymentContract()
        self.assertIsInstance(dc.resource, ResourceContract)
        self.assertIsInstance(dc.control, ControlContract)
        self.assertIsInstance(dc.runtime, RuntimeContract)
        self.assertTrue(dc.runtime.finite_required)

    def test_contracts_are_immutable(self) -> None:
        dc = DeploymentContract()
        with self.assertRaises(Exception):
            dc.resource.cycles_max = 999  # type: ignore[misc]  # frozen dataclass — проверяем запрет записи


class TestVerificationResult(unittest.TestCase):
    def test_passed_without_violations(self) -> None:
        res = VerificationResult(passed=True, violations=(), metrics={"iae": 1.5})
        self.assertTrue(res.passed)
        self.assertEqual(res.violations, ())
        self.assertEqual(res.metrics["iae"], 1.5)

    def test_compact_flattens_for_csv(self) -> None:
        res = VerificationResult(
            passed=False,
            violations=("iae", "overshoot"),
            metrics={"iae": 9.5, "overshoot": 1.7},
        )
        row = res.compact()
        self.assertEqual(row["verification_passed"], 0.0)
        self.assertEqual(row["violations"], "iae,overshoot")
        self.assertEqual(row["iae"], 9.5)
        self.assertEqual(row["overshoot"], 1.7)
        # компакт-строка должна состоять только из чисел и строк — иначе CSV-экспорт сломается
        for value in row.values():
            self.assertIsInstance(value, (int, float, str))

    def test_compact_passed_is_one(self) -> None:
        row = VerificationResult(passed=True, violations=(), metrics={}).compact()
        self.assertEqual(row["verification_passed"], 1.0)
        self.assertEqual(row["violations"], "")


class TestCustomContracts(unittest.TestCase):
    def test_override_single_field(self) -> None:
        rc = ResourceContract(cycles_max=90)
        self.assertEqual(rc.cycles_max, 90)
        # остальные поля берутся из умолчаний
        self.assertEqual(rc.ram_bytes_max, ResourceContract().ram_bytes_max)

    def test_deployment_with_custom_resource(self) -> None:
        dc = DeploymentContract(resource=ResourceContract(ram_bytes_max=64))
        self.assertEqual(dc.resource.ram_bytes_max, 64)
        self.assertEqual(dc.control, ControlContract())

    def test_control_contract_bounds_are_ordered(self) -> None:
        cc = ControlContract()
        # допуски должны быть неотрицательными, иначе вердикт всегда «провал»
        self.assertGreaterEqual(cc.iae_max, 0.0)
        self.assertGreaterEqual(cc.overshoot_max, 0.0)
        self.assertGreaterEqual(cc.final_error_max, 0.0)
        self.assertGreater(cc.max_abs_y, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

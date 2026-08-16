# Python Prototype — RACS + DSO (Resource-Aware Controller Synthesis)

Python-прототип DSO-ControlLab: синтез регуляторов генетическим программированием
с учётом ресурсных ограничений реального времени (Cortex-M4).

## Файлы

| Файл | Описание |
|---|---|
| `racs2.py` | Ядро: expression-tree GP, PID/LQR/MPC, 5 plants, Cortex-M4 cycle model (калиброван) |
| `dso.py` | Memory Planner (арены), Resource Contract, Determinism Analyzer (variance-centric) |
| `racs_dso.py` | Интегрированный бенчмарк: PID/GP/GP-R/DSO-GP + variance-анализ |
| `racs_nsga.py` | NSGA-II multi-objective: (variance-proxy, IAE, ресурсы), Pareto front |
| `racs_codegen/` | Tree → C → clang/ld.lld → QEMU Cortex-M4 → реальные циклы (WCET) |

## Ключевые результаты

- **NSGA-II vs PID на non-minimum phase: IAE 1.06 vs 27.85 (26×)** при DetVar=88.7, 12 cycles
- Модель циклов **калибрована по QEMU** (VDIV=14, min/max=10, sin=128, sqrt=140) — совпадение ±7%
- DSO-критерий качества — **variance** (детерминизм), IAE вторичен

## Запуск

```bash
python3 racs_nsga.py            # NSGA-II бенчмарк (все 5 plants)
python3 racs_dso.py             # scalarized DSO-GP бенчмарк
PYTHONPATH=. python3 racs_codegen/codegen.py   # верификация циклов на QEMU
```

Подробности: `RACS_DSO_RESULTS.md`

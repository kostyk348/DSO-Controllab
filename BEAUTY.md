# BEAUTY.md — что осталось за владельцем

Аудит (`repo_polish_audit.py`, прогон 14.10.2026) по этому репозиторию:

```
DSO-Controllab   ★4  2026-09-13 → нет CHANGELOG.md; нет ни одного релиза/тега
```

У воркера нет задачи ставить GitHub-метаданные, теги и релизы. Ниже — только то, чего не сделать
из репозитория, с готовыми командами. Запускать их имеет смысл **после** публикации локального
коммита из этой ветки.

## 1. Состояние CI (проверено через GitHub API 15.10.2026)

**CI был красным у всех прогонов** — последние 5 запусков `failure` (по 12–15 с):

```
gh run list -R kostyk348/DSO-Controllab --limit 5
# completed  failure  docs(python_proto): finalize RACS+DSO status …  34774337768  12s
# completed  failure  feat(python_proto): DSO streaming compiler …    34773127806  14s
```

Причина — в `ci.yml` ставился только `numpy`, а шаг запускал `python3 -m pytest tests/`:

```
/opt/hostedtoolcache/Python/3.11.16/x64/bin/python3: No module named pytest
##[error]Process completed with exit code 1.
```

В этом коммите причина устранена (`pip install numpy pytest` + шаг smoke-прогона README).
После пуша стоит убедиться, что workflow зелёный **до** постановки тега.

## 2. Первый релиз / тег

Тег и релиз предлагаются только при зелёном CI — иначе зелёный бейдж врёт.

```bash
# после пуша и подтверждённо зелёного CI на main:
git tag -a v0.1.0 -m "v0.1.0 — first public research snapshot: PID/LQR/MPC/DSO bench + GP synthesis in C with Ada SPARK verification"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-file CHANGELOG.md
```

`CHANGELOG.md` в этом коммите уже в формате Keep a Changelog — годится как notes-file.

## 3. Метаданные (описание, темы)

Уже заполнены, менять не обязательно:

* описание: `Research prototype for Deterministic Systems Optimization - verified execution plans and
  resource contracts for control systems. Apache 2.0.`
* темы (15): `c`, `cli`, `control-systems`, `deterministic`, `dso`, `library`, `lqr`, `mpc`,
  `optimization`, `pid`, `python`, `research`, `resource-contracts`, `scada`, `verification`.

Если хочется обновить описание/темы одной командой (идемпотентно, лишние темы не удаляет):

```bash
gh repo edit kostyk348/DSO-Controllab \
  --description "Reproducible research bench for Deterministic Systems Optimization: PID/LQR/MPC/DSO comparison, GP controller synthesis in C11, Ada SPARK/GNATprove verification. Apache-2.0." \
  --add-topic genetic-programming \
  --add-topic ada-spark \
  --add-topic gnatprove
```

`genetic-programming` / `ada-spark` / `gnatprove` в списке тем отсутствуют, а по содержимому
репозитория (GP-движок и формальная верификация) они уместны.

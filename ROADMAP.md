# DSO Roadmap

This roadmap keeps the project publishable at every stage.

## Phase 0: ControlLab Prototype

Status: current.

Goal:

- prove the DSO framing on a measurable control benchmark
- compare ordinary runtime-heavy strategies with pre-verified plans
- make contracts visible

Done:

- random control worlds
- PID/LQR/MPC/DSO comparison
- resource/control/runtime/deployment contracts
- pre-deployment verification
- JSON reports

Next:

- CSV export
- plots
- GP controller synthesis
- multiple verification seeds per world
- per-violation statistics

## Phase 1: GitHub Release

Goal:

- make the repo understandable to strangers
- present DSO as an idea, not just code

Tasks:

- add `VISION.md`
- add `ROADMAP.md`
- add `CONTRACTS.md`
- add article draft
- add `.gitignore`
- add license
- add reproducible command examples
- include one small JSON result file
- avoid committing large generated artifacts unless needed

## Phase 2: libdso

Goal:

- create a small user-space deterministic runtime for normal programs

Core features:

- arena allocator
- bounded queues
- job graph
- static schedule
- resource contract structs
- trace recorder
- simple verifier API

Minimal API sketch:

```c
dso_arena arena = dso_arena_create(buffer, size);
dso_graph graph = dso_graph_create(&arena);

dso_task_id a = dso_graph_task(graph, "load", load_fn);
dso_task_id b = dso_graph_task(graph, "process", process_fn);
dso_graph_dep(graph, a, b);

dso_contract contract = {
    .memory_bytes = 65536,
    .max_tasks = 64,
    .max_step_us = 1000,
};

dso_plan plan = dso_compile(graph, contract);
dso_verify(plan);
dso_execute(plan);
```

## Phase 3: DSO Planner

Goal:

- separate planning logic from runtime

Planner responsibilities:

- lifetime analysis
- memory layout
- task ordering
- budget checks
- schedule generation
- contract violation reports

Output:

```text
plan.json
trace.json
memory.map
schedule.txt
```

## Phase 4: Linux User-Space Integration

Goal:

- apply DSO principles to PC programs without kernel work

Possible integrations:

- CPU pinning through `sched_setaffinity`
- cgroups for CPU/memory limits
- `mlock`/`mlockall` for memory predictability
- `io_uring` for planned IO
- eBPF for tracing real execution
- perf counters for validation

Deliverable:

```text
dso-run --contract app.contract.json -- ./my_program
```

## Phase 5: Demonstrators

Build demos that show DSO across domains:

- `dso-controllab`: control systems
- `dso-ai`: small inference graph
- `dso-game-loop`: deterministic frame graph
- `dso-embedded`: STM32/ESP32-style static plan
- `dso-linux-run`: user-space Linux contract runner

## Phase 6: Linux Scheduler Research

Goal:

- investigate scheduler support for contract-aware processes

Do not start here.

Research question:

> Can Linux schedule processes better when they expose verified execution graphs and resource contracts?

Possible shape:

- user-space prototype first
- eBPF measurement
- custom scheduler class only after evidence
- compare against CFS, RT scheduling and cgroups

## Near-Term Priority

The next best engineering move:

```text
DSO-ControlLab
  -> polish GitHub repo
  -> write public article
  -> add plots and CSV
  -> start libdso
```

This is more credible than jumping directly to an OS.

# DSO Vision

DSO means Deterministic Systems Optimization.

The central claim is simple:

> A program should make as many execution decisions as possible before runtime. Runtime should execute a verified plan, not constantly rediscover the plan.

DSO is not a replacement for operating systems, compilers, data-oriented design, real-time systems or schedulers. It is an engineering discipline that connects them:

- describe work as dataflow and resource lifetimes
- make ownership and budgets explicit
- plan memory, scheduling and synchronization before execution
- verify plans before deployment
- keep runtime small, predictable and observable

## Why Apply DSO To Normal Programs?

Most software is written as if runtime uncertainty is normal:

- allocate when needed
- schedule when work appears
- discover dependencies dynamically
- grow queues dynamically
- synchronize dynamically
- retry dynamically
- recover dynamically

This is flexible, but it makes performance and behavior harder to reason about.

DSO asks a different question:

> Which parts of this program can be turned into a known plan?

Examples:

- fixed-size arenas instead of unbounded allocation
- known task graphs instead of dynamic worker chaos
- bounded queues instead of unbounded buffering
- known memory lifetimes instead of hidden ownership
- explicit resource budgets instead of best effort
- verified execution slots instead of ad hoc runtime adaptation

## The General DSO Model

```text
Program description
  -> contracts
  -> analysis
  -> planning
  -> verification
  -> execution plan
  -> simple runtime
```

The runtime is intentionally boring:

```text
for task in plan:
    execute(task)
```

The intelligence belongs in the compiler, planner, verifier and tooling.

## Where DSO Fits

### 1. Control Systems

ControlLab is the first proof point:

- compare PID/LQR/MPC/DSO
- generate random worlds
- verify plans before deployment
- measure quality and resource behavior

This domain is good because determinism matters and results are measurable.

### 2. User-Space Runtime For PC Programs

This is the best next step.

Build `libdso`, a small user-space runtime:

- arenas
- bounded queues
- job graph
- static scheduling
- resource contracts
- trace export
- verification hooks

This lets normal Linux programs use DSO without touching the kernel.

### 3. Linux Integration

After `libdso`, integrate with Linux from user space:

- CPU affinity
- `sched_setaffinity`
- cgroups
- `SCHED_FIFO` / `SCHED_RR` where appropriate
- `io_uring` for bounded IO plans
- eBPF tracing for validation

This is more realistic than starting with a kernel scheduler.

### 4. DSO Scheduler For Linux

A kernel scheduler is a later research target, not the first product.

The interesting idea is not "replace CFS". The interesting idea is:

> schedule contract-verified execution graphs differently from best-effort processes.

A future Linux DSO scheduler could treat a process as:

```text
contract + graph + budget + deadline + memory profile
```

instead of only:

```text
thread + priority + vruntime
```

But this only makes sense after user-space DSO proves that programs can expose useful contracts.

### 5. AI Inference

Neural networks are already graphs. DSO can add:

- memory lifetime planning
- arena allocation
- deterministic execution graphs
- streaming plans
- budgeted inference

This is a strong long-term direction.

### 6. Games And Engines

Game frames are natural DSO units:

```text
input -> simulation -> animation -> render graph -> streaming -> output
```

DSO can make frame time, memory and streaming more explicit.

### 7. Embedded And Robotics

This is one of the strongest domains:

- limited memory
- hard latency
- expensive failure
- sensors and actuators already form pipelines

DSO contracts map naturally to embedded constraints.

## Recommended Direction

Do not start with an OS or kernel scheduler.

The strongest path is:

```text
1. DSO-ControlLab
2. libdso user-space runtime
3. DSO Planner
4. Linux integration layer
5. AI/Game/Embedded demos
6. kernel scheduler research
```

This path creates usable artifacts at every step.

## One-Sentence Positioning

DSO is an engineering methodology and runtime architecture for turning dynamic software behavior into verified execution plans with explicit resource contracts.

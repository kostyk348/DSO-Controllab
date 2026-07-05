# DSO Contracts

In DSO, a contract is not documentation. It is an executable boundary between planning and deployment.

The DSO pipeline is:

```text
Candidate Plan
  -> static resource analysis
  -> simulation verification
  -> contract check
  -> deployment
```

Runtime should not discover whether a plan is acceptable. Runtime should receive a plan that has already passed its contracts.

## Contract Layers

### 1. Resource Contract

This describes the execution budget:

- maximum cycles per control step
- maximum RAM bytes
- maximum WCET
- maximum jitter

Example:

```text
cycles <= 180
RAM <= 96 bytes
WCET <= 8 us
jitter <= 0.8 us
```

This answers: can the controller physically run inside the target machine budget?

### 2. Control Contract

This describes the allowed control behavior:

- maximum IAE
- maximum overshoot
- maximum absolute output
- maximum final error

This answers: does the controller behave acceptably on the plant family?

### 3. Runtime Contract

This describes runtime safety:

- no NaN / Inf
- bounded saturation fraction
- bounded state/output excursions

This answers: can the runtime execute this plan without emergency behavior becoming normal behavior?

### 4. Deployment Contract

This is the combined contract:

```text
DeploymentContract =
  ResourceContract
  + ControlContract
  + RuntimeContract
```

A plan is deployable only when every layer passes.

## Why This Matters

Classic runtime-heavy systems often say:

```text
Run it, observe what happens, adapt online.
```

DSO says:

```text
Search offline, verify offline, deploy only a bounded plan.
```

The important difference is not that DSO forbids adaptation. The difference is that adaptation must enter through a contract:

```text
event -> queue -> bounded planner -> verified plan slot -> runtime
```

The runtime does not become an unbounded decision machine.

## Current Prototype

The current ControlLab implementation has:

- typed contract objects in `src/dso_controllab/contracts.py`
- verification logic in `src/dso_controllab/verifier.py`
- DSO plan selection that rejects candidates with contract violations
- report fields for contract and verification status

This is still a prototype. The next step is to make the contracts stronger:

- verify across several noise seeds per world
- add plant-family contracts instead of single-world contracts
- add formal bounds for state and control signals
- export per-violation statistics
- use the same contract model for GP-generated controllers

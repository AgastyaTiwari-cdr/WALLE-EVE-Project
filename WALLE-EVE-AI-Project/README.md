# WALL-E & EVE: Autonomous Signal Search Agent

A Python AI/ML-style simulation in which WALL-E searches a partially observable grid to locate EVE. The agent estimates EVE's hidden location from noisy sensor readings, explores for clues, decides when to scan or recharge, and records an auditable trace of its decisions.

> **Project scope:** This is a model-based Bayesian search agent with heuristic information-seeking planning. It is not a trained neural network or an exact POMDP solver. The optional LLM benchmark is a separate, API-backed comparison and is not required to run the core project.

## What someone viewing this GitHub repository will see

- The source code for the simulator, Bayesian belief filter, autonomous agent, multi-agent mode, and benchmarking tools.
- A sample mission configuration (`config.json`) with a 10×10 map, sensor noise, clues, and decoy signal sources.
- Sample JSON outputs showing the mission trace, multi-agent run, and planner benchmark.
- An ASCII grid in the terminal when a normal mission runs, plus a final mission report.
- Automated tests that check core behavior.

This repository is a **command-line simulation**, not a hosted website or graphical game. Someone who clones it can run the simulation and see the grid and JSON results in their terminal. No physical robot is controlled.

## Features

- **Hidden-state simulation:** the simulator stores EVE's true coordinate; the agent receives observations rather than direct access to the true state.
- **Bayesian localization:** a grid belief distribution is updated using a Gaussian signal likelihood and clue evidence, with log-space handling for numerical stability.
- **Information-seeking exploration:** candidate destinations are ranked using belief, expected signal-discrimination value, novelty, and travel cost. This is a heuristic, not globally optimal planning.
- **Evidence collection:** clues are revealed only when an agent reaches their cell and are incorporated into the belief update.
- **Noisy sensors and decoys:** configurable noise, dropouts, outliers, bias, and decoy emitters test robustness.
- **Active sensing:** a scan consumes battery and gives a lower-noise reading, making it a trade-off.
- **Resource management:** movement and scanning consume battery; recharging is capped at capacity; missions stop at terminal conditions.
- **Cooperative multi-agent mode:** agents share a belief/evidence map and visited-cell information to coordinate their search.
- **Planner benchmarks:** compares a simple classical MAP baseline with the information-seeking agent over deterministic seeds.
- **Optional real LLM planner:** API-backed experimental comparison; requires a valid API key and may incur cost. No key is included in this repository.
- **Audit trail:** JSON logs include actions, observations, battery, belief estimates, entropy, rationale, and final mission status.

## Quick start

Python 3.10+ is recommended.

### 1. Create an environment and install dependencies

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run one WALL-E mission

```bash
python main.py --config config.json
```

Save the report somewhere else:

```bash
python main.py --config config.json --output results/mission.json
```

### 3. Run the cooperative multi-agent simulation

```bash
python -m src.multi_agent --config config.json --agents 2 --output multi_agent_report.json
```

### 4. Benchmark the planners

```bash
python -m src.benchmark --config config.json --seeds 20 --output planner_benchmark.json
```

The benchmark reports success rate, mean steps, remaining battery, clue discovery, and scan counts. Results depend on configuration and seeds. The committed benchmark is an example run, not a universal performance claim.

### 5. Run the tests

```bash
python -m pip install pytest
python -m pytest -q
```

## Optional: compare against a real LLM

The LLM option uses an external API and is optional. It requires internet access, a valid API key, and may incur usage charges.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
python -m src.benchmark --config config.json --seeds 5 --include-llm --llm-model gpt-4o-mini --output planner_benchmark_llm.json
```

macOS/Linux:

```bash
export OPENAI_API_KEY="your-api-key"
python -m src.benchmark --config config.json --seeds 5 --include-llm --llm-model gpt-4o-mini --output planner_benchmark_llm.json
```

Never commit your API key. `.env.example` is only a placeholder; `.gitignore` excludes local `.env` files. The LLM is provided only with agent-side observations and belief summaries, not the simulator's hidden EVE coordinate. If the API call fails or returns an unusable response, the code can fall back to local planning.

## Repository layout

```text
WALLE-EVE-AI-Project/
├── main.py                         # Single-agent mission entry point
├── config.json                     # Reproducible sample mission
├── requirements.txt                # Runtime dependency list
├── README.md                       # Documentation and commands
├── LICENSE
├── .gitignore
├── .env.example                    # Optional LLM key placeholder
├── src/
│   ├── agent.py                    # Information-seeking agent
│   ├── belief_filter.py            # Bayesian belief updates
│   ├── environment.py              # Hidden-state simulator and sensors
│   ├── schema.py                   # Actions, coordinates, observations
│   ├── multi_agent.py              # Cooperative team search
│   └── benchmark.py                # Baseline, agent and optional LLM comparison
├── tests/
│   └── test_project.py             # Automated tests
├── mission_traces_and_report.json  # Example single-agent output
├── multi_agent_report.json         # Example multi-agent output
└── planner_benchmark.json          # Example benchmark output
```

## Configuration notes

- `environment.grid_shape` is `[width, height]`; coordinates are `[x, y]` with `(0, 0)` at the lower-left.
- `environment.true_eve_position` is ground truth used by the simulator. The agent's observation does not expose it, but it is visible in the public sample configuration so runs can be reproduced and evaluated.
- `sensor_noise_sigma` must be positive. A scan uses a lower noise level for the next observation.
- `environment.sensor_model` supports `dropout_probability`, `outlier_probability`, `outlier_scale`, and `bias`.
- `decoys` can add false signal emitters. They contribute to sensor readings but are not exposed as known target locations to the agent.
- `clues` are revealed when an agent reaches their coordinates.
- `random_seed` makes a mission reproducible; use different seeds to evaluate robustness.

## Limitations and honest interpretation

- The planner is heuristic and is not guaranteed to find EVE in every mission or under every budget.
- The belief model assumes a single target source; decoys deliberately violate that simplifying assumption and test robustness.
- The multi-agent implementation is a shared-state simulation, not a distributed networking protocol.
- The LLM benchmark is optional and must be run with an API key to produce live LLM results.
- Benchmark figures are configuration-dependent. For a defensible evaluation, use many seeds, report the settings, and compare against clearly described baselines.
- The sample configuration exposes ground-truth EVE coordinates to readers of the repository for reproducibility; this is not exposed through the agent's observation API during simulation.

## Assignment mapping

| Requirement | Implementation |
|---|---|
| Configurable simulated environment | `config.json`, `src/environment.py` |
| Autonomous movement, scanning, recharge | `src/agent.py`, `src/environment.py` |
| Belief updates and replanning | `src/belief_filter.py`, `src/agent.py` |
| Full action trace | `main.py` and JSON mission report |
| Final mission report | `mission_traces_and_report.json` |
| Visualization | ASCII terminal grid |
| Advanced noisy sensors/decoys | `src/environment.py`, `config.json` |
| Multi-agent coordination | `src/multi_agent.py` |
| Planner comparison | `src/benchmark.py` |

## License

MIT. See [LICENSE](LICENSE).

# complex_sytem_simulation

## Bornholdt / Ising Spin Model of Market Dynamics on Networks

## Repo layout 

### `code/`
Subpackages:
- `bornholdt_2d/` — baseline 2D lattice model + runners + plotting  
- `bornholdt_network/` — network version (ER / WS / BA) + runners + plotting  
- `bornholdt_heterogeneity/` — lattice model with quenched heterogeneous \(\alpha_i\)  
- `tests/` — pytest suites for baseline + heterogeneity invariants & reproducibility  

### `data/` (outputs + reference market series)  
- Simulation outputs (CSV):  
  `intermediate.csv`, `intermediate_heterogeneity.csv`, `network_run.csv`  
- Reference market data (CSV):  
  `BTC-USD.csv`, `GSPC.csv`  

### `paper_figures/` and `paper_figures_network/`
Saved PNGs used in the write-up/paper-style presentation:  
- returns time series  
- CCDFs of \(|r|\)  
- volatility autocorrelations  
- chartist fraction plots (strategy composition vs time/volatility)  

### Outputs
- **Simulation CSVs** in `data/` are the inputs to the plotting scripts.  
- **PNGs** in `paper_figures/` and `paper_figures_network/` are the generated figures (lattice + network variants).  

### Bonuses
- You have roughly 5-10% assert statements (currently under)
- pytest works and succeeds (need to add network pytest)
- Structured as a module (“__init__.py”; e.g. “pip install .” works) (Good)
- Documentation generated from docstrings (Good)

###

### Main idea
Financial markets exhibit robust fat-tailed returns, volatility clustering, and intermittent regime shifts that are not well captured by equilibrium or fully rational-agent models.  
In this project and code nbase we study a spin-based market model (Bornholdt-type), where traders are binary spins that interact via:  

1) Local imitation (herding): agents align with their neighbors  
2) Global contrarian pressure: agents are pushed to act against the crowd when the market becomes too one-sided  

We extend the baseline 2D-lattice model in two directions central to complex systems:  
- Interaction topology: replace lattice neighborhoods with complex networks (Erdős–Rényi, Watts–Strogatz, Barabási–Albert) while keeping average degree comparable.  
- Agent heterogeneity: replace a single global contrarian strength with agent-specific sensitivities.  
 
Our guiding question:  
- Can market-like complexity emerge from minimal microscopic rules, and how do topology and heterogeneity shape regime stability and volatility clustering?  

## Model (Bornholdt-type two-scale Ising market)

### Agent state
Each agent \(i = 1,\dots,N\) has:  
- Decision spin \(S_i(t) \in \{+1,-1\}\) (buy / sell)  
- Strategy spin \(C_i(t) \in \{+1,-1\}\) (fundamentalist / chartist label)  

### Magnetization
\[  
M(t) = \frac{1}{N}\sum_{j=1}^N S_j(t)  
\]  

### Local field (two scales: local herding + global contrarian)  
Baseline (homogeneous \(\alpha\)):  
\[
h_i(t) = \sum_{j=1}^N J_{ij} S_j(t)\;-\;\alpha\, C_i(t)\, M(t)  
\]  

**Network vs lattice coupling**  
- Lattice: \(J_{ij}=J\) only for nearest neighbors  
- Network: \(J_{ij}=J A_{ij}\), where \(A_{ij}\) is the adjacency matrix  

### Spin update (heat-bath / logistic rule)
Let \(\beta = 1/T\) be inverse temperature:  
\[
\mathbb{P}\big(S_i(t{+}1)=+1\;\big|\;h_i(t)\big)
= \frac{1}{1+\exp\{-2\beta h_i(t)\}}
\]  
(and \(S_i(t{+}1)=-1\) otherwise)  

### Strategy switching
\[
C_i(t{+}1) = -C_i(t) \quad \text{if}\quad C_i(t)\, S_i(t)\, M(t) < 0
\]
(otherwise \(C_i(t{+}1)=C_i(t)\))  

### Returns (from magnetization)
We define log returns from changes in market polarization:  
\[
r(t)=\ln |M(t)|-\ln |M(t{-}1)|
\]  

## Extensions

### 1) Agent heterogeneity (random \(\alpha_i\))
Instead of a single \(\alpha\), assign each agent a sensitivity \(\alpha_i\) drawn from a fixed distribution at initialization.  
\[
h_i(t) = \sum_{j=1}^N J_{ij} S_j(t)\;-\;\alpha_i\, C_i(t)\, M(t)
\]  
Interpretation:  
- large \(\alpha_i\): strongly contrarian agents  
- small \(\alpha_i\): weakly sensitive agents
  
### 2) Network topology
We simulate the same dynamics on multiple topologies (with matched average degree where possible):  
- 2D lattice (baseline)  
- Erdős–Rényi (random)  
- Watts–Strogatz (small-world)  
- Barabási–Albert (scale-free)  

## Observable  
We measure:  
- Return time series \(r(t)\) (intermittency / bursts)  
- Tail behavior via CCDF of \(|r|\) (fat tails)  
- Volatility clustering via autocorrelation of \(|r|\) (or \(r^2\)) over lags  
- Regime persistence and switching via magnetization behavior and visual snapshots of spin domains  

## Hypotheses and outcomes

Note: These are primarily qualitative checks from simulation outputs. We did not run exhaustive parameter sweeps or finite-size scaling, so “reject/accept” is interpreted as supported vs. not supported.  

### H1 — Criticality & Frustration
**Claim:** There exists a transition between ordered (herding-dominated) and disordered market regimes as coupling strength or temperature is varied, characterized by peaks in susceptibility and volatility clustering.
**Outcome:** 

**Verdict:** Do not reject (qualitative support).  

### H2 — Topology matters  
**Claim:** At fixed average degree, interaction topology (2D lattice vs. Erdős–Rényi vs. Watts–Strogatz vs. Barabási–Albert) shifts the location and sharpness of the critical region and alters the strength of volatility clustering and regime persistence.
**Outcome:**   
Across WS / ER / BA we still reproduce the same qualitative stylized facts (fat-tailed \(|r|\), volatility clustering), which suggests the mechanism is not lattice-specific.  
At the same time, topology appears to modulate the dynamics (e.g., the sharpness/strength of bursts, persistence patterns, tail extent), indicating that who-interacts-with-whom matters.  

**Verdict:** Do not reject. A full critical region shift test requires systematic sweeps.  

### H3 — Agent heterogeneity  
**Claim:** Heterogeneity in contrarian sensitivity \(\alpha_i\) changes market dynamics and stylized facts.  
**Outcome:** 

**Verdict:** Do not reject.  

## Conclusions
- Market-like complexity can emerge from minimal microscopic rules: local herding + global contrarian pressure is sufficient to produce intermittent dynamics, fat tails, and volatility clustering.  
- Networks can capture the same market dynamics as the lattice model, suggesting the phenomenon is robust to interaction structure.  
- Heterogeneity and network topology act as modulators rather than primary drivers: they tune regime switching speed, persistence, and the strength/sharpness of volatility clustering, without destroying the core facts.  

### Limitations & outlook
Parameter exploration was limited. Next steps:  
- systematic sweeps over temperature/coupling and network parameters,  
- finite-size scaling to better locate phase transitions and universality classes,  
- implement endogenous feedback to test self-organized regime switching more directly.  

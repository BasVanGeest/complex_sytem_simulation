# Complex System Simulation

## Bornholdt / Ising Spin Model of Market Dynamics on Networks and with Agent Heterogeneity

**Authors:** Edoardo Putignano, George Petropoulos, Jiawei Li, Bas van Geest

----------

## Overview and Scope

This project implements, reproduces, and extends the seminal spin-based market models introduced by **Stefan Bornholdt (2001)** and extended by **Takuya Yamano (2002)**. Our primary goal is twofold:

1.  **Baseline reproduction**: faithfully reproduce the qualitative stylized facts reported in the reference papers—fat-tailed return distributions, volatility clustering, intermittent regime switching—using the original 2D lattice Bornholdt model.
    
2.  **Systematic extensions**: extend the same microscopic dynamics to (i) heterogeneous agents and (ii) complex interaction topologies, and **compare all extensions against the baseline using the same observables and figures** as in the reference literature.
    

Throughout, we deliberately follow the model definitions, update rules, and observables used by Bornholdt and Yamano, in order to make direct, apples-to-apples comparisons possible.

----------

## Research Questions and Hypotheses

The project is guided by the following research questions, formulated to mirror the structure and claims of the reference literature while remaining accessible to numerical investigation:

-   **H1: Criticality and frustration.** Does the competition between local herding and global contrarian pressure induce a transition between frozen, intermittent, and disordered market regimes as temperature or coupling strength is varied?
    
-   **H2: Role of interaction topology.** At fixed average degree, how does changing the interaction topology (2D lattice vs. Erdős–Rényi, Watts–Strogatz, and Barabási–Albert networks) affect regime stability, intermittency, and volatility clustering?
    
-   **H3: Agent heterogeneity.** How does heterogeneity in agents’ contrarian sensitivity (lpha_i) influence regime persistence, intermittency, and the emergence of stylized market facts?
    

----------

## Repository Layout

### `code/`

Python modules implementing the model and experiments:

-   `bornholdt_2d/` — baseline 2D lattice model, runners, and plotting scripts
    
-   `bornholdt_network/` — network-based models (Erdős–Rényi, Watts–Strogatz, Barabási–Albert)
    
-   `bornholdt_heterogeneity/` — lattice model with fixed agent heterogeneity (assigned randomly at initialization)
    
-   `tests/` — pytest suites testing invariants, reproducibility, and basic consistency
    

Each subpackage (`bornholdt_2d/`, `bornholdt_heterogeneity/`, `bornholdt_network/`, `tests/`) contains its own README file with detailed instructions on how to run the corresponding simulations or tests, including recommended numbers of Monte Carlo sweeps for reliable results.

### `data/`

Simulation outputs and reference datasets:

-   **Simulation CSVs**: `lattice_data_results_100000.csv`, `heterogeneity_data_results_100000.csv`, `ER_data_results_100000.csv`, `WS_data_results_100000.csv`, `BA_data_results_100000.csv`
    
-   **Reference market data**: `BTC-USD.csv`, `GSPC.csv`, to visualize the same robust features in real market settings
    

### `results/`, `paper_figures/`, `paper_figures_network/`

Generated figures used in the report-style analysis:

-   return time series
    
-   CCDFs of absolute returns |r|
    
-   volatility autocorrelations
    
-   chartist–fundamentalist composition over time
    

### Project structure and software practices

In line with the course development guidelines, the repository is organized as a proper Python package and includes several software-engineering best practices:

-   modular code structure with `__init__.py` files
    
-   extensive use of docstrings and inline comments
    
-   inline `assert` statements to enforce invariants and catch silent failures
    
-   automated tests that can be executed with `pytest` and run successfully
    
-   a reproducible environment specified via `requirements.txt`
    

These features support reproducibility, maintainability, and ease of extension.

----------

## Scientific Motivation

Real financial markets display robust empirical regularities—fat-tailed returns, clustered volatility, and intermittent regime shifts—that are not well explained by equilibrium or fully rational-agent models.

The Bornholdt spin model demonstrates that **minimal microscopic rules**, combining:

1.  **local herding** (neighbor imitation), and
    
2.  **global contrarian pressure** (frustration with market-wide imbalance),
    

are sufficient to generate these stylized facts.

Our project investigates how robust this mechanism is when:

-   the interaction topology is changed from a lattice to a complex network, and
    
-   agents are no longer identical, but differ in their sensitivity to global market pressure.
    

----------

## Baseline Model: Bornholdt Two-Scale Ising Market

### Agent State

Each agent i carries two binary variables:

-   **Decision spin**: (S_i(t) \in {+1, -1}), representing buy/sell pressure.
    
-   **Strategy spin**: (C_i(t) \in {+1, -1}), interpreted as:
    
    -   (+1): _fundamentalist_ (contrarian to the majority)
        
    -   (-1): _chartist_ (trend-following)
        

### Local Field (Baseline)

Following Bornholdt (2001), the effective local field acting on agent i is:

$$  
h_i(t) = \sum_j J_{ij} S_j(t) - \alpha , C_i(t) , \frac{1}{N} \sum_{j=1}^N S_j(t)  
$$

where:

-   (J_{ij} = J) for nearest neighbors (2D lattice), zero otherwise,
    
-   the first term induces **local ferromagnetic ordering** (herding),
    
-   the second term couples agents to the **global magnetization** (market imbalance), generating frustration across scales.
    

### Spin Update Rule

Agents update asynchronously using heat-bath dynamics:

$$  
P(S_i(t+1) = +1) = \frac{1}{1 + e^{-2 \beta h_i(t)}}  
$$

where (\beta = 1/T) plays the role of an inverse temperature controlling stochasticity.

### Strategy Switching

Strategy spins evolve according to the Bornholdt rule:

$$  
C_i(t+1) = -C_i(t) \quad \text{if} \quad \alpha S_i(t) C_i(t) \sum_j S_j(t) < 0  
$$

This mechanism captures adaptive switching between chartist and fundamentalist behavior depending on market conditions.

### Market Observables

The global magnetization

$$  
M(t) = \frac{1}{N} \sum_i S_i(t)  
$$

is interpreted as a price-like quantity. Returns are defined as:

$$  
r(t) = \ln |M(t)| - \ln |M(t-1)|  
$$

This definition follows both Bornholdt (2001) and Yamano (2002), enabling direct comparison of:

-   return distributions,
    
-   volatility clustering,
    
-   regime persistence.
    

----------

## Model Extensions

### 1) Agent Heterogeneity

To relax the assumption of identical agents, we assign each agent a **agent-specific contrarian strength (fixed over time)** (\alpha_i).

-   (\alpha_i \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)), drawn at initialization
    
-   **Constraint:** (\alpha_i > 0)
    

Negative (\alpha_i) would correspond to agents that _reinforce_ global trends rather than opposing them, implying qualitatively different and economically unrealistic strategies. For this reason, the distribution is truncated to strictly positive values.

This extension allows us to study how diversity in agent sensitivity affects volatility clustering, regime switching, and tail behavior.

### 2) Network Topology

We replace the 2D lattice with interaction networks while keeping average degree comparable:

-   Erdős–Rényi (random)
    
-   Watts–Strogatz (small-world)
    
-   Barabási–Albert (scale-free)
    

The same microscopic update rules and observables are used, enabling direct comparison with the lattice baseline and the reference figures.

----------

## Observables and Comparison Strategy

For both baseline and extensions we analyze:

-   return time series (intermittency)
    
-   CCDF of |r| (fat tails)
    
-   autocorrelation of |r| (volatility clustering)
    
-   magnetization dynamics and strategy composition
    

All extensions are evaluated **using the same figures and diagnostics employed in the reference papers**, ensuring consistency.

----------

## Conclusions

### Baseline dynamics and reproduction

Our baseline implementation closely follows the formulations of Bornholdt (2001) and Yamano (2002) and successfully reproduces their central qualitative findings. In particular, the interplay between **local herding** and **global contrarian pressure** generates a frustrated dynamics characterized by long-lived metastable ordered phases that are intermittently interrupted by rapid, system-wide reorganization events.

As in the reference papers, the temperature (or noise level) plays a crucial role in shaping the collective dynamics. For very low temperatures, stochasticity is strongly suppressed and the system rapidly freezes into ordered configurations, with agents locked into stable clusters and virtually no macroscopic activity. At sufficiently high temperatures, noise dominates the interactions: local order is destroyed, coherent clusters fail to form, and the dynamics becomes effectively chaotic, with no persistent large-scale patterns.

Between these extremes lies an intermediate regime at temperatures below but not too far from the critical temperature $T_c$. In this regime, the system exhibits **intermittent dynamics**: long-lived metastable ordered phases—characterized by coherent agent clusters—are suddenly interrupted by rapid, system-wide reorganization events. Small microscopic fluctuations can then trigger macroscopic rearrangements of the market state, leading to bursts of activity and volatility reminiscent of near-critical behavior.

In this regime, we observe:

-   **Fat-tailed return distributions**, evidenced by heavy-tailed CCDFs of |r|, consistent with scale-free (power-law-like) behavior over an intermediate range;
    
-   **Volatility clustering**, where periods of high volatility are followed by further high-volatility periods, despite agents being memoryless and updates being stochastic;
    
-   **Intermittent dynamics**, with alternating phases of relative calm and sudden bursts of activity.
    

These emergent properties arise without fine-tuning and are direct consequences of frustration across interaction scales.

### Agent heterogeneity

Introducing fixed (time-independent) heterogeneity in agents’ contrarian sensitivity does not qualitatively alter the macroscopic behavior of the system. Instead, heterogeneity acts as a smoothing mechanism: it broadens transition regions, modifies persistence times of metastable phases, and slightly reshapes burst statistics, while preserving intermittent dynamics, fat-tailed returns, and volatility clustering. This indicates that the Bornholdt mechanism is robust to realistic levels of agent diversity.

### Network topology

Replacing the regular lattice with complex interaction networks likewise preserves the core stylized facts. Across Erdős–Rényi, Watts–Strogatz, and Barabási–Albert networks, we consistently observe heavy-tailed returns and clustered volatility. Network topology primarily modulates quantitative aspects of the dynamics—such as the sharpness of bursts, the duration of ordered phases, and the strength of correlations—rather than fundamentally changing the qualitative behavior.

### Synthesis

Taken together, our results support the view that **market-like complexity can emerge robustly from minimal microscopic rules**. Frustration between local imitation and global contrarian forces is the primary driver of intermittency and volatility clustering, while heterogeneity and topology shape how these effects manifest in time. The persistence of stylized facts across extensions suggests that the mechanism is not lattice-specific, nor dependent on agent homogeneity, but instead reflects a generic feature of frustrated multi-agent systems operating near criticality.

----------

## Limitations and Outlook

-   Parameter exploration was limited; no exhaustive sweeps or finite-size scaling were performed.
    
-   Future work could include:
    
    -   systematic phase-diagram mapping,
        
    -   finite-size scaling and universality analysis
        

----------
## GenAI Usage Statement

Generative AI tools were used as **technical support** during development, mainly for debugging, code organization, file handling (CSV input/output), plotting utilities, and documentation improvements. All core modeling decisions, extensions, and scientific analysis were developed independently.

A detailed and transparent description of how GenAI was used can be found in the accompanying file **`AI_usage_report.md`**.

----------

## References

-   Stefan Bornholdt (2001), _Expectation bubbles in a spin model of markets_, arXiv:cond-mat/0105224
    
-   Takuya Yamano (2002), _Bornholdt’s spin model of a market dynamics in high dimensions_, arXiv:cond-mat/0110279
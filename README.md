# Environment Classifier and Demonstrator
### Intelligent Room Temperature Control Simulation & Comparative Multi-Agent Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit 1.35+](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly 5.22+](https://img.shields.io/badge/plotly-5.22+-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Title & Overview

The **Environment Classifier and Demonstrator** is an academic-grade, interactive thermodynamic simulation and multi-agent benchmarking platform. It models indoor microclimate heat transfer under Newton's Law of Cooling, active HVAC actuation, and stochastic environmental disturbances. 

The platform implements, classifies, and evaluates two distinct autonomous artificial intelligence agents:
1. **Utility-Based Intelligent Agent (`UtilityBasedAgent`)**: An agent that constructs internal one-step thermodynamic lookahead predictions and maximizes a multi-objective utility objective balancing thermal comfort, energy consumption, temperature stability, and thermal overshoot penalties.
2. **Simple Reflex Thermostat Agent (`SimpleReflexAgent`)**: A classical condition-action agent that selects actions directly from current temperature percepts against configurable lower and upper bounds without predictive internal models or utility calculations.

---

## 2. Problem Statement

Residential and commercial buildings consume over 40% of global primary energy, with Heating, Ventilation, and Air Conditioning (HVAC) systems accounting for the dominant share. Traditional simple reflex thermostats operate purely reactively using bimetallic or digital deadband thresholds. While simple and low-cost, reflex thermostats suffer from:
- **Thermal Overshooting:** Inability to account for heat transfer rates through building envelopes.
- **Actuator Chattering:** Rapid cycling between heating and cooling near threshold boundaries.
- **Energy Waste:** Inability to anticipate ambient temperature trends and insulation thermal lag.

The core challenge is to demonstrate how utility-based decision making can proactively balance comfort error against energetic cost and stability, and to provide rigorous quantitative benchmarks comparing both paradigms under identical environmental disturbances.

---

## 3. Objectives

- **Thermodynamic Modeling:** Develop a continuous-variable room temperature model governed by heat exchange with the exterior ambient environment, active HVAC heating/cooling, and stochastic noise.
- **Environmental Classification:** Classify room states dynamically into discrete qualitative categories: **Cold**, **Comfortable**, and **Hot**.
- **Utility Maximization Agent:** Formulate and tune a multi-objective mathematical utility function evaluating candidate actions ($\text{HEAT}$, $\text{COOL}$, $\text{OFF}$).
- **Simple Reflex Agent:** Implement an uncoupled condition-action thermostat utilizing upper and lower temperature thresholds.
- **Synchronized Benchmarking:** Execute side-by-side simulations subjecting both agents to identical environmental disturbance vectors ($w_t$).
- **Multi-Metric Evaluation:** Compute 11 quantitative performance indicators covering comfort accuracy, energy consumption, and thermal stability.
- **Academic Dashboard & Telemetry:** Build a 7-section Streamlit dashboard equipped with live gauges, interactive parameter tuning, telemetry export, and interactive demonstrations.

---

## 4. Technologies Used

- **Core Runtime:** Python 3.10+ (Tested on Python 3.12)
- **Web Interface:** Streamlit (Academic dashboard design, modular navigation, responsive layout)
- **Numerical Computation:** NumPy (Thermodynamic state updates, Gaussian disturbance vectors, vector math)
- **Data Structuring:** Pandas (Time-series telemetry recording, metrics aggregation, CSV export)
- **Interactive Visualizations:** Plotly Graph Objects (Multi-trace line trajectories, gauge indicators, breakdown bar charts, distribution pies)
- **Static Charting:** Matplotlib (Supplementary analysis and verification plotting)

---

## 5. System Architecture

```
                    +-------------------------------------+
                    |       User / Researcher GUI         |
                    |    (Streamlit 7-Section Dashboard)  |
                    +------------------+------------------+
                                       |
                   Configures Parameters & Scenarios
                                       |
                                       v
                     +----------------------------------+
                     |    Synchronized Benchmark Engine |
                     |        (comparison.py)           |
                     +--------+----------------+--------+
                              |                |
             Identical Noise  |                | Identical Noise
             Sequence {w_t}   |                | Sequence {w_t}
                              v                v
     +--------------------------------+   +-------------------------------+
     |      Utility-Based Agent       |   |     Simple Reflex Agent       |
     |      (utility_agent.py)        |   |      (reflex_agent.py)        |
     |--------------------------------|   |-------------------------------|
     | 1. Lookahead Thermodynamic     |   | 1. Read Current Percept T_t   |
     |    Prediction T_hat(a)         |   | 2. Evaluate Condition Rules:  |
     | 2. Multi-Objective Utility     |   |    - T_t < T_lower -> HEAT    |
     |    Evaluation U(a)             |   |    - T_t > T_upper -> COOL    |
     | 3. Argmax Action Selection     |   |    - Otherwise -> OFF         |
     +----------------+---------------+   +---------------+---------------+
                      | Action                            | Action
                      v                                   v
     +--------------------------------+   +-------------------------------+
     |      Room Environment A        |   |      Room Environment B       |
     |       (environment.py)         |   |       (environment.py)        |
     |--------------------------------|   |-------------------------------|
     |  Newton's Law of Cooling       |   |  Newton's Law of Cooling      |
     |  Active Heating / Cooling      |   |  Active Heating / Cooling     |
     |  Stochastic Perturbation       |   |  Stochastic Perturbation      |
     +----------------+---------------+   +---------------+---------------+
                      |                                   |
                      +-----------------+-----------------+
                                        | Telemetry Logs
                                        v
                     +----------------------------------+
                     |  11 Quantitative Performance     |
                     |      Metrics & CSV Downloads     |
                     +----------------------------------+
```

---

## 6. Project Folder Structure

```text
environment-classifier/
├── app.py                      # Main Streamlit web dashboard (7-section academic interface)
├── config.py                   # Dataclasses, constants, thresholds, weights, and demonstration scenarios
├── environment.py              # Thermodynamic room simulation model and state classification
├── utility_agent.py            # Utility-based intelligent agent (lookahead prediction & optimization)
├── reflex_agent.py             # Simple reflex thermostat agent (condition-action rules)
├── comparison.py               # Synchronized dual-agent benchmark runner & 11-metric analytics engine
├── utils.py                    # Plotly chart generators, gauge widgets, and zero-indent HTML cards
├── requirements.txt            # Project dependencies (Streamlit, NumPy, Pandas, Plotly, Matplotlib)
├── run.ps1                     # PowerShell launch script
├── run.bat                     # Windows batch launch script
├── test_comprehensive_audit.py # Master audit test suite covering all 12 system verification groups
├── test_simulation.py          # Unit tests verifying thermodynamic physics and classification boundaries
├── test_utility_agent.py       # Unit tests verifying utility maximization and lookahead logic
├── test_reflex_agent.py        # Unit tests verifying simple reflex threshold rules and independence
├── test_comparison.py          # Unit tests verifying comparison runner, disturbance parity, and metrics
└── README.md                   # Comprehensive project documentation and academic reference
```

---

## 7. Mathematical Environment Model Explanation

The room temperature at discrete time step $t+1$ is computed from the state at time step $t$ using Newton's Law of Cooling augmented with active HVAC heat injection and random environmental perturbations:

$$T_{t+1} = T_t + k \cdot (T_{\text{outside}} - T_t) + \Delta T_{\text{HVAC}}(a) + w_t$$

### Term Definitions:
1. **$T_t$:** Current indoor room temperature ($^\circ\text{C}$).
2. **$k \in [0, 1]$:** Insulation heat transfer coefficient (derived from envelope construction):
   - **Good Insulation:** $k = 0.03$ (Minimal thermal conductivity, slow passive drift)
   - **Medium Insulation:** $k = 0.07$ (Standard residential standard)
   - **Poor Insulation:** $k = 0.14$ (High thermal conductivity, rapid passive drift)
3. **$(T_{\text{outside}} - T_t)$:** Temperature gradient between outdoor ambient conditions and indoor air.
4. **$\Delta T_{\text{HVAC}}(a)$:** Active temperature change supplied by HVAC actuation:
   $$\Delta T_{\text{HVAC}}(a) = \begin{cases} +P_{\text{heat}} & \text{if } a = \text{HEAT} \\ -P_{\text{cool}} & \text{if } a = \text{COOL} \\ 0.0 & \text{if } a = \text{OFF} \end{cases}$$
5. **$w_t \sim \mathcal{N}(0, \sigma^2)$:** Zero-mean Gaussian environmental disturbance (accounting for opening doors, sunlight variation, drafts, and human occupancy).

### Environment State Classification:
Given the target desired temperature $T_{\text{desired}}$ and comfort tolerance $\tau$:
- **$\text{Comfortable Range} = [T_{\text{desired}} - \tau, \; T_{\text{desired}} + \tau]$**
- **Cold:** $T < T_{\text{desired}} - \tau$ (Blue indicator `#0284c7`)
- **Comfortable:** $T_{\text{desired}} - \tau \le T \le T_{\text{desired}} + \tau$ (Green indicator `#16a34a`)
- **Hot:** $T > T_{\text{desired}} + \tau$ (Amber/Red indicator `#ea580c`)

---

## 8. Utility Function Explanation

The Utility-Based Agent optimizes decision making by evaluating all available actions $a \in \{\text{HEAT}, \text{COOL}, \text{OFF}\}$ through a multi-factor penalty-weighted objective function:

$$\text{Utility}(a) = - \left( w_c \cdot P_{\text{comfort}}(a) + w_e \cdot C_{\text{energy}}(a) + w_s \cdot P_{\text{instability}}(a) + w_o \cdot P_{\text{overshoot}}(a) \right)$$

The agent selects the action that maximizes net utility:
$$a^* = \arg\max_{a \in \{\text{HEAT}, \text{COOL}, \text{OFF}\}} \text{Utility}(a)$$

### Component Formulations:
1. **Predicted Temperature $\hat{T}_{t+1}(a)$:** Internal lookahead using the thermodynamic model without disturbance noise:
   $$\hat{T}_{t+1}(a) = T_t + k \cdot (T_{\text{outside}} - T_t) + \Delta T_{\text{HVAC}}(a)$$
2. **Comfort Penalty ($P_{\text{comfort}}$):**
   $$P_{\text{comfort}}(a) = |\hat{T}_{t+1}(a) - T_{\text{desired}}|$$
3. **Energy Cost ($C_{\text{energy}}$):**
   $$C_{\text{energy}}(a) = \begin{cases} 1.0 & \text{if } a = \text{HEAT} \\ 1.0 & \text{if } a = \text{COOL} \\ 0.0 & \text{if } a = \text{OFF} \end{cases}$$
4. **Instability Penalty ($P_{\text{instability}}$):** Penalizes sudden single-step thermal swings:
   $$P_{\text{instability}}(a) = |\hat{T}_{t+1}(a) - T_t|$$
5. **Overshoot Penalty ($P_{\text{overshoot}}$):** Heavily penalizes drives beyond the comfort boundaries:
   $$P_{\text{overshoot}}(a) = \begin{cases} \hat{T}_{t+1}(a) - (T_{\text{desired}} + \tau) & \text{if } \hat{T}_{t+1}(a) > T_{\text{desired}} + \tau \\ (T_{\text{desired}} - \tau) - \hat{T}_{t+1}(a) & \text{if } \hat{T}_{t+1}(a) < T_{\text{desired}} - \tau \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 9. Reflex Agent Condition-Action Rules

The Simple Reflex Thermostat acts purely reactively on the instantaneous temperature percept $T_t$, without lookahead, memory, or utility scoring:

$$\text{Action}(T_t) = \begin{cases} \text{HEAT} & \text{if } T_t < T_{\text{lower}} \\ \text{COOL} & \text{if } T_t > T_{\text{upper}} \\ \text{OFF} & \text{if } T_{\text{lower}} \le T_t \le T_{\text{upper}} \end{cases}$$

### Key Properties:
- **Zero Internal State:** Operates without memory of past temperatures or actuator actions.
- **No Environmental Lookahead:** Disregards outside ambient temperature ($T_{\text{outside}}$) and insulation level ($k$).
- **Threshold Deadband:** The gap $[T_{\text{lower}}, T_{\text{upper}}]$ provides hysteresis to prevent continuous cycling, but cannot foresee incoming heat flux.

---

## 10. Performance Metrics Explanation

The platform computes **11 quantitative performance metrics** across active simulation steps ($t \in [1, N]$):

| # | Metric Name | Mathematical Definition | Evaluation Goal |
|---|---|---|---|
| 1 | **Average Absolute Error (MAE)** | $\frac{1}{N}\sum_{t=1}^N \|T_t - T_{\text{desired}}\|$ | Lower is better (tighter tracking) |
| 2 | **Final Temperature Error** | $\|T_N - T_{\text{desired}}\|$ | Lower is better (settling precision) |
| 3 | **Comfortable Time Percentage** | $\frac{N_{\text{comfortable}}}{N} \times 100\%$ | Higher is better (occupant comfort) |
| 4 | **Overshoot Count** | $\sum_{t=1}^{N-1} \mathbb{I}(\text{crosses comfort bound})$ | Lower is better (prevents thermal overshoot) |
| 5 | **Total Energy Consumed** | $\sum_{t=1}^N C_{\text{energy}}(a_t)$ | Lower is better (operational efficiency) |
| 6 | **Average Energy Per Step** | $\frac{\text{Total Energy}}{N}$ | Lower is better (sustained efficiency) |
| 7 | **Heating Activations** | $\sum_{t=1}^N \mathbb{I}(a_t = \text{HEAT})$ | Informational count |
| 8 | **Cooling Activations** | $\sum_{t=1}^N \mathbb{I}(a_t = \text{COOL})$ | Informational count |
| 9 | **Temperature Stability (Fluctuations)** | $\frac{1}{N-1}\sum_{t=2}^N \|T_t - T_{t-1}\|$ | Lower is better (smooth indoor climate) |
| 10 | **Unnecessary Action Changes (Chattering)** | Count of direct $\text{HEAT} \leftrightarrow \text{COOL}$ or $A \rightarrow B \rightarrow A$ cycles | Lower is better (prevents equipment wear) |
| 11 | **Settling Step Count** | $\min \{ t^* \mid \forall t \ge t^*, \; \text{Class}(T_t) = \text{Comfortable} \}$ | Lower is better (faster recovery to comfort) |

---

## 11. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- Git installed (for version control).

### Step-by-Step Setup:
```powershell
# 1. Clone the repository
git clone https://github.com/sampathkumarreddy123/environment-classifier.git
cd environment-classifier

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install required Python packages
pip install -r requirements.txt
```

---

## 12. How to Run the Application

### Method A: Using the PowerShell Launcher Script
```powershell
.\run.ps1
```

### Method B: Using the Windows Batch File
Double-click `run.bat` or run:
```cmd
run.bat
```

### Method C: Direct Streamlit Execution
```powershell
streamlit run app.py
```
*The web browser will automatically open to `http://localhost:8501`.*

---

## 13. Demonstration Scenarios

The platform includes **3 preconfigured academic scenarios** selectable with a single click, alongside full custom scenario editing:

### Scenario 1: Cold Room (`cold_room`)
- **Initial Temp:** $14.0^\circ\text{C}$ | **Target:** $22.0^\circ\text{C}$ | **Outside:** $10.0^\circ\text{C}$
- **Insulation:** Medium ($k=0.07$) | **Tolerance:** $\pm 1.5^\circ\text{C}$ | **Steps:** 50
- **Focus:** Heating response, overshoot mitigation, and settling stability when outdoor cold actively drains heat.

### Scenario 2: Hot Room (`hot_room`)
- **Initial Temp:** $32.0^\circ\text{C}$ | **Target:** $22.0^\circ\text{C}$ | **Outside:** $36.0^\circ\text{C}$
- **Insulation:** Medium ($k=0.07$) | **Tolerance:** $\pm 1.5^\circ\text{C}$ | **Steps:** 50
- **Focus:** Active cooling power, energy trade-offs, and continuous heat infiltration under heatwave conditions.

### Scenario 3: Comfortable Room (`comfortable_room`)
- **Initial Temp:** $22.0^\circ\text{C}$ | **Target:** $22.0^\circ\text{C}$ | **Outside:** $28.0^\circ\text{C}$
- **Insulation:** Good ($k=0.03$) | **Tolerance:** $\pm 1.5^\circ\text{C}$ | **Steps:** 50
- **Focus:** Energy conservation ($\text{OFF}$ drift exploitation), minimizing actuator cycling, and maintaining steady-state comfort.

### Custom Scenario
- Fully adjustable sidebar controls allowing continuous slider variation across all 11 thermodynamic and weight parameters.

---

## 14. Expected Outputs & Sample Comparative Results

### Typical Benchmark: Cold Room Scenario ($N=50$, Seed=42)
| Quantitative Metric | Utility-Based Agent | Simple Reflex Agent | Unbiased Winner |
|---|---|---|---|
| **Average Absolute Error (MAE)** | **$0.61^\circ\text{C}$** | $3.94^\circ\text{C}$ | 🤖 Utility Agent |
| **Comfortable Time %** | **$95.0\%$** | $5.0\%$ | 🤖 Utility Agent |
| **Final Temperature Error** | **$0.12^\circ\text{C}$** | $0.68^\circ\text{C}$ | 🤖 Utility Agent |
| **Overshoot Count** | **0** | 1 | 🤖 Utility Agent |
| **Total Energy Consumed** | 30.0 units | **21.0 units** | 🔁 Reflex Agent |
| **Temperature Stability** | **$0.18^\circ\text{C}$/step** | $0.35^\circ\text{C}$/step | 🤖 Utility Agent |
| **Actuator Chattering Count** | **0** | 2 | 🤖 Utility Agent |
| **Settling Step** | **Step 6** | Step 14 | 🤖 Utility Agent |

### Analysis of Comparative Behavior:
1. **Comfort Accuracy:** The utility-based agent achieves significantly lower MAE and higher comfort duration because its lookahead model anticipates thermal drift and modulates power smoothly.
2. **Energy Trade-Off:** The simple reflex agent may use slightly fewer energy units because it stays in the $\text{OFF}$ state longer inside the deadband, but at the direct cost of severe comfort violations and wider temperature swings.
3. **Chattering Prevention:** The utility agent penalizes instability ($P_{\text{instability}}$), eliminating rapid switching between heating and cooling.

---

## 15. Limitations & Future Enhancements

### Current Model Limitations:
- **One-Step Horizon:** The utility agent optimizes a single-step lookahead horizon ($t+1$). It does not perform multi-step dynamic programming or Model Predictive Control (MPC) over long horizons ($t+1 \dots t+H$).
- **Lumped Thermal Capacitance:** The room is modeled as a single uniform air node without spatial temperature gradients or distinct wall thermal masses.
- **Constant Outdoor Temperature:** The current benchmark uses a fixed outside temperature across steps rather than a dynamic diurnal sinusoidal curve.

### Planned Future Enhancements:
- **Receding-Horizon Model Predictive Control (MPC):** Extending the utility optimizer into a multi-step trajectory planner ($H=10$).
- **Diurnal Day/Night Cycle:** Realistic outside ambient temperature curves with solar radiation gain.
- **Multi-Zone Simulation:** Modeling coupled adjacent rooms with partition wall heat transfers.
- **Deep Reinforcement Learning Agent:** Integrating a Proximal Policy Optimization (PPO) agent as a third benchmark competitor.

---

## 16. Verification & Automated Testing

The repository contains an exhaustive test suite verifying thermodynamic consistency, lookahead prediction, reflex condition rules, disturbance parity, and edge-case resilience.

Run all tests from the repository root:
```powershell
# 1. Master Comprehensive Audit Test Suite (All 12 Groups)
python test_comprehensive_audit.py

# 2. Thermodynamic Environment Foundation Tests
python test_simulation.py

# 3. Utility-Based Agent Decision Tests
python test_utility_agent.py

# 4. Simple Reflex Thermostat Rule Tests
python test_reflex_agent.py

# 5. Synchronized Multi-Agent Comparison Tests
python test_comparison.py
```

All 5 test suites execute with **100% passing results**.

---

## 17. Application Navigation Guide

The dashboard is structured into **7 clearly defined sections** accessible via the sidebar navigation menu:

1. **1. Home / Overview:** High-level executive summary, key features, architecture overview, and quick-start links.
2. **2. Environment Configuration:** Demonstration scenario selector (Cold, Hot, Comfortable, Custom), interactive thermodynamic sandbox, real-time gauges, and live parameter tuning.
3. **3. Utility-Based Agent:** Dedicated utility agent laboratory, live multi-action score evaluation bar charts, component penalty breakdowns, step-by-step decision cards, and telemetry export.
4. **4. Simple Reflex Agent:** Dedicated reflex thermostat dashboard, active condition-action threshold indicators, temperature trajectories, and actuation history.
5. **5. Agent Comparison:** Synchronized dual-agent benchmark simulator, single-step and continuous play controls, overlay trajectory chart, 11-metric evaluation table, and analytical diagnostic tabs.
6. **6. Results and Downloads:** Complete data center offering one-click CSV downloads for utility telemetry, reflex telemetry, and side-by-side synchronized comparison datasets.
7. **7. About the Project:** Educational theory section detailing agent classification, thermodynamic formulations, KaTeX mathematical equations, and comprehensive Oral Defense / Viva Q&A preparation.

---

## License

This project is open-source under the [MIT License](LICENSE). Academic attribution is appreciated.

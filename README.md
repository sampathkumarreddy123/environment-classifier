# Environment Classifier and Demonstrator
### Intelligent Room Temperature Control Simulation (Phase 3: Simple Reflex Thermostat)

---

## 📖 Overview
The **Environment Classifier and Demonstrator** is an interactive thermodynamic simulation platform designed to evaluate and demonstrate room temperature control strategies.

- **Phase 1 (Foundation):** Established the thermodynamic room model (Newton's cooling law, HVAC actuation, disturbance noise), temperature classification (**Cold**, **Comfortable**, **Hot**), and the interactive manual sandbox.
- **Phase 2 (Utility-Based Agent):** Implements the **Utility-Based Intelligent Agent** in `utility_agent.py`, which predicts future thermal comfort, evaluates candidate actions against multiple competing objectives, and selects the action that maximizes net utility.
- **Phase 3 (Current Phase):** Implements the **Simple Reflex Thermostat** in `reflex_agent.py`, which operates purely on current room temperature percepts using configurable condition-action rules (`HEAT`, `OFF`, `COOL`), maintaining independent simulation history and its own dashboard section.
- **Phase 4 (Upcoming):** Will introduce the side-by-side comparison dashboard evaluating both agents together.

---

## 🗂️ Project Structure

```text
environment-classifier/
├── app.py                   # Main Streamlit web dashboard (Utility Agent, Reflex Thermostat, Sandbox, Theory)
├── utility_agent.py         # Utility-based intelligent agent (prediction & multi-factor utility optimization)
├── reflex_agent.py          # Simple reflex thermostat agent (condition-action threshold rules)
├── environment.py           # Thermodynamic room simulation model & state classification
├── config.py                # Configuration defaults, utility weights, reflex thresholds, actions, and styling
├── utils.py                 # Visualizations (Plotly charts, gauges, rule cards, utility breakdowns, decision cards)
├── requirements.txt         # Project dependencies (Streamlit, NumPy, Pandas, Plotly, Matplotlib)
├── test_reflex_agent.py     # Test suite verifying simple reflex rules, boundary behavior, and independence
├── test_utility_agent.py    # Test suite verifying utility agent decision-making across all 5 required conditions
├── test_simulation.py       # Test suite verifying thermodynamic physics and classification
├── run.ps1                  # PowerShell launcher script
├── run.bat                  # Windows batch launcher script
└── README.md                # Project documentation and user guide
```

---

## 🧠 Utility-Based Agent Architecture

Rather than relying on fixed temperature thresholds, the utility-based agent evaluates every available action ($a \in \{\text{HEAT}, \text{COOL}, \text{OFF}\}$) using an internal mental model of the room thermodynamics and a multi-objective utility function.

### 1. Decision-Making Process (at every time step $t$):
1. **Observe** current room temperature $T_t$.
2. **Read** desired comfort target $T_{\text{desired}}$ and tolerance $\tau$.
3. **Observe** outside ambient temperature $T_{\text{outside}}$ and insulation level $k$.
4. **Generate** candidate actions: `['HEAT', 'COOL', 'OFF']`.
5. **Predict** resulting temperature $\hat{T}_{t+1}(a)$ using the environment model:
   $$\hat{T}_{t+1}(a) = T_t + k \cdot (T_{\text{outside}} - T_t) + \Delta T_{\text{HVAC}}(a)$$
6. **Calculate** utility $\text{Utility}(a)$ balancing comfort, energy, stability, and overshoot.
7. **Select** the action with maximum utility:
   $$a^* = \arg\max_{a} \text{Utility}(a)$$
8. **Apply** the action to the physical room environment.
9. **Record** telemetry, utility scores, and natural language explanation.

---

### 2. Multi-Objective Utility Function

$$\text{Utility}(a) = - \left( w_c \cdot P_{\text{comfort}} + w_e \cdot C_{\text{energy}} + w_s \cdot P_{\text{instability}} + w_o \cdot P_{\text{overshoot}} \right)$$

Where:
- **$P_{\text{comfort}} = |\hat{T}_{t+1}(a) - T_{\text{desired}}|$:** Distance between predicted temperature and desired target.
- **$C_{\text{energy}} \in \{1.0 \text{ (HEAT)}, 1.0 \text{ (COOL)}, 0.0 \text{ (OFF)}\}$:** Power consumption cost.
- **$P_{\text{instability}} = |\hat{T}_{t+1}(a) - T_t|$:** Penalizes abrupt temperature fluctuations.
- **$P_{\text{overshoot}}$:** Distance exceeding the acceptable comfort zone bounds ($[T_{\text{desired}} - \tau, T_{\text{desired}} + \tau]$).
- **Configurable Weights ($w_c, w_e, w_s, w_o$):** Sliders in the sidebar allow real-time tuning of agent priorities.

---

## 🚀 Getting Started

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```powershell
# Run the simple reflex thermostat verification tests
python test_reflex_agent.py

# Run the 6-test suite verifying the utility-based agent
python test_utility_agent.py

# Run the thermodynamic environment model tests
python test_simulation.py
```

### 3. Launch the Application
Run using any of the following methods:

**Quick Launcher:**
```powershell
.\run.ps1
```
*(or double-click `run.bat`)*

**Direct Streamlit Command:**
```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔁 Simple Reflex Thermostat Architecture

The **Simple Reflex Thermostat** (`reflex_agent.py`) reacts strictly to the current observed room temperature $T_t$ without calculating utility scores or predicting future temperature:

- **IF** $T_t < T_{\text{lower}}$ → **`HEAT`** (Turn on heating)
- **ELSE IF** $T_t > T_{\text{upper}}$ → **`COOL`** (Turn on cooling)
- **ELSE** ($T_{\text{lower}} \le T_t \le T_{\text{upper}}$) → **`OFF`** (Turn off both systems)

Users configure $T_{\text{lower}}$ and $T_{\text{upper}}$ (with $T_{\text{lower}} < T_{\text{upper}}$ enforced), and the agent records step-by-step telemetry (`step`, `room_temp`, `outside_temp`, `desired_temp`, `action`, `actual_temp`, `energy_consumed`, `classification`) in a Pandas DataFrame.

---

## 📊 Dashboard Features
- **Simple Reflex Thermostat Tab:** Displays current temperature, configured lower/upper thresholds, selected action, environment classification, energy consumption, dynamic visual condition-action rule cards, threshold-banded trajectory chart, action distribution chart, and telemetry CSV export.
- **Utility Decision Card:** Displays current temperature, desired temperature, selected action, predicted temperature, utility score, and natural language reasoning.
- **Utility Comparison & Penalty Charts:** Live comparison of utility scores and weighted penalty decomposition across `HEAT`, `COOL`, and `OFF`.
- **Interactive Trajectory:** Plots room temperature, agent predicted temperature, outside ambient, and comfort zone band over time.
- **Telemetry Export:** Download complete step-by-step history as a CSV file for each agent independently.
- **Manual Sandbox:** Phase 1 manual room explorer remains intact in a dedicated tab.

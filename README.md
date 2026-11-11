# Environment Classifier and Demonstrator
### Intelligent Room Temperature Control Simulation (Phase 1 Foundation)

---

## 📖 Overview
The **Environment Classifier and Demonstrator** is an interactive thermodynamic simulation platform designed to evaluate and compare intelligent control strategies. 

In **Phase 1 (Foundation)**, the project establishes:
1. A continuous thermodynamic room environment model with heat exchange, insulation factors, HVAC actuation, and environmental disturbances.
2. A temperature classification system categorizing room states into **Cold**, **Comfortable**, and **Hot**.
3. A modern Streamlit dashboard featuring interactive configuration controls, live temperature gauges, comparative bar charts, and trajectory history plots.

In **Phase 2**, this foundation will be used to compare two intelligent agents:
- **Utility-Based Agent**: Optimizes actions by maximizing a multi-objective utility function (thermal comfort vs. energy consumption).
- **Simple Reflex Thermostat**: Operates on fixed threshold rules with hysteresis.

---

## 🗂️ Project Structure

```text
environment-classifier/
├── app.py                # Main Streamlit web application & dashboard UI
├── environment.py        # Thermodynamic room simulation model & state management
├── config.py             # Simulation constants, default limits, color schemes, and actions
├── utils.py              # Visualizations (Plotly gauge, line charts, bar comparisons, status cards)
├── requirements.txt      # Project dependencies (Streamlit, NumPy, Pandas, Plotly, Matplotlib)
├── test_simulation.py    # Automated test suite for simulation physics and classifier logic
└── README.md             # Project documentation and user guide
```

---

## 🧮 Thermodynamic Mathematical Model

The room temperature at time step $t + 1$ is modeled according to continuous thermodynamic principles:

$$T_{t+1} = T_t + \Delta T_{\text{ambient}} + \Delta T_{\text{HVAC}} + \epsilon_t$$

Where:
1. **Ambient Heat Transfer (Newton's Law of Cooling):**
   $$\Delta T_{\text{ambient}} = k \cdot (T_{\text{outside}} - T_t)$$
   $k$ is the insulation heat-loss coefficient:
   - **Poor:** $k = 0.14$ (rapid heat transfer)
   - **Medium:** $k = 0.07$ (standard double-glazed insulation)
   - **Good:** $k = 0.025$ (high-performance thermal insulation)

2. **HVAC Actuation:**
   $$\Delta T_{\text{HVAC}} = \begin{cases} +P_{\text{heater}} & \text{if Action = HEAT} \\ -P_{\text{cooler}} & \text{if Action = COOL} \\ 0 & \text{if Action = OFF} \end{cases}$$

3. **Stochastic Disturbance:**
   $$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
   Where $\sigma$ is the disturbance level (modeling opened doors, solar gain fluctuations, occupant body heat).

---

## 🏷️ Temperature Classification

The room state is classified relative to the target desired temperature $T_{\text{desired}}$ and comfort tolerance $\tau$ (default $\pm 1.5^\circ\text{C}$):

- ❄️ **Cold:** $T_t < T_{\text{desired}} - \tau$
- 🌿 **Comfortable:** $T_{\text{desired}} - \tau \le T_t \le T_{\text{desired}} + \tau$
- 🔥 **Hot:** $T_t > T_{\text{desired}} + \tau$

---

## 🚀 Getting Started

### 1. Requirements
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify that the thermodynamic model and classification logic pass all checks:
```bash
python test_simulation.py
```

### 4. Launch the Web Application
You can run the application directly using any of the following methods:

**Option A (Quick Launcher Script):**
```powershell
.\run.ps1
```
or double-click `run.bat` in File Explorer.

**Option B (Standard Streamlit Command):**
If you open a new PowerShell terminal:
```powershell
streamlit run app.py
```

If using your current PowerShell session, refresh the PATH first:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎮 Interactive Controls
- **Sidebar**: Adjust initial temperature, comfort target, outside temperature, room insulation, HVAC power, and disturbance noise.
- **Interactive Stepping**: Step through the simulation step-by-step or run a full batch run with manual HVAC actions (OFF, HEAT, COOL) to observe thermal inertia and response curves.
- **Data Export**: Download the simulated trajectory data as a CSV file for analysis.

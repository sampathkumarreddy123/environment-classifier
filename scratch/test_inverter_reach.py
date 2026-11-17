import sys
sys.path.insert(0, ".")
from environment import RoomEnvironment
from utility_agent import UtilityBasedAgent
from reflex_agent import SimpleReflexAgent
from comparison import ComparisonConfig, ComparisonRunner

def boosted_step(self, action="OFF", disturbance=None):
    action = action.upper()
    if action not in ["OFF", "HEAT", "COOL"]:
        action = "OFF"

    delta_ambient = self.k_loss * (self.outside_temp - self.current_temp)
    effective_occupants = self.num_people if self.people_present else 0
    delta_occupancy = effective_occupants * self.heat_gain_per_person * self.room_scale

    dist = abs(self.current_temp - self.desired_temp)

    if action == "HEAT":
        delta_hvac = self.heater_power * self.room_scale
        if self.current_temp < self.desired_temp and not self.target_reached:
            rem_steps = max(1, 6 - self.current_step)
            needed_net = min(dist, max(self.heater_power * self.room_scale, dist / rem_steps))
            needed_heat = needed_net + max(0.0, -delta_ambient)
            delta_hvac = max(delta_hvac, needed_heat)
    elif action == "COOL":
        delta_hvac = -self.cooling_power * self.room_scale
        if self.current_temp > self.desired_temp and not self.target_reached:
            rem_steps = max(1, 6 - self.current_step)
            needed_net = min(dist, max(self.cooling_power * self.room_scale, dist / rem_steps))
            needed_cool = needed_net + max(0.0, delta_ambient) + delta_occupancy
            delta_hvac = -max(abs(delta_hvac), needed_cool)
    else:
        delta_hvac = 0.0

    if disturbance is not None:
        applied_disturbance = float(disturbance)
    elif self.disturbance_level > 0.0:
        applied_disturbance = float(self._rng.normal(0.0, self.disturbance_level))
    else:
        applied_disturbance = 0.0

    total_delta = delta_ambient + delta_hvac + delta_occupancy + applied_disturbance
    new_temp = self.current_temp + total_delta

    if self.people_present:
        if self.target_reached:
            new_temp = self.desired_temp
            total_delta = 0.0
            delta_ambient = 0.0
            delta_hvac = 0.0
            delta_occupancy = 0.0
            applied_disturbance = 0.0
        else:
            reached = False
            if (self.current_temp <= self.desired_temp and new_temp >= self.desired_temp) or \
               (self.current_temp >= self.desired_temp and new_temp <= self.desired_temp) or \
               abs(new_temp - self.desired_temp) <= 0.2:
                reached = True

            if reached:
                self.target_reached = True
                new_temp = self.desired_temp
                total_delta = 0.0
                delta_ambient = 0.0
                delta_hvac = 0.0
                delta_occupancy = 0.0
                applied_disturbance = 0.0

    self.current_step += 1
    self.current_temp = new_temp
    self._record_step(
        step=self.current_step,
        temp=self.current_temp,
        action=action,
        delta_ambient=delta_ambient,
        delta_hvac=delta_hvac,
        delta_occupancy=delta_occupancy,
        disturbance=applied_disturbance,
    )
    return self.get_state()

RoomEnvironment.step = boosted_step

cfg = ComparisonConfig(
    initial_temp=40.0,
    desired_temp=22.0,
    outside_temp=45.0,
    people_present=True,
    num_people=3,
    room_size=25.0,
    max_steps=15,
)
runner = ComparisonRunner(cfg)
for step in range(15):
    runner.step_once()
    u = runner.utility_records[-1]
    r = runner.reflex_records[-1]
    print(f"Step {step+1}: Utility={u['actual_temp']:.2f}°C [{u['action']}] | Reflex={r['actual_temp']:.2f}°C [{r['action']}]")

res = runner.evaluate()
print("\n--- RESULTS ---")
print("Utility Energy:", res["utility_metrics"].total_energy)
print("Reflex Energy:", res["reflex_metrics"].total_energy)
print("Overall Winner:", res["summary"]["overall_winner"])
for row in res["comparison_df"].to_dict(orient="records"):
    print(f"{row['Performance Metric']}: Utility={row['Utility-Based Agent']} | Reflex={row['Simple Reflex Agent']} | Winner={row['Better Performing Agent']}")

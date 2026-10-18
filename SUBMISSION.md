# Team Assigned Team 1

## Members
| Name | Matric number | Email |
| --- | --- | --- |
| [Murugesan Sakthitharan] | [U2622972C] | [sakthith001@e.ntu.edu.sg] |
| [Dylan D Mehta] | [U2521007L] | [dylan004@e.ntu.edu.sg] |
| [Liu Tian Ren] | [U2522751D] | [tliu027@e.ntu.edu.sg] |

## Approach
The driver follows a precomputed map-derived trajectory with Pure Pursuit and
speed-adaptive lookahead. Speed planning previews upcoming curvature, limits
lateral acceleration, and anticipates braking. Command acceleration and steering
rate are limited. A LiDAR guard considers both forward clearance and the intended
pursuit direction, with gap recovery for obstruction handling.

We selected settings using official race results and retained a clean ten-lap
configuration. We rejected several changes, including predictive steering and
more aggressive preview braking, after their measured performance or reliability
was worse. Inactive earlier experiments remain in the code with their defaults off.

## How it uses the inputs
- LiDAR: inflated obstacle ranges, forward and intended-direction sectors, recovery.
- Odometry: simulator ground-truth position, orientation and speed for tracking.
- Map: supplied icra26 map and centreline inform an offline trajectory with clearance
  constraints. No manual corner-coordinate steering or illegal cone-gap shortcuts.

## Results on our own machine

Three official local referee evaluations are included, unedited,
under `results/submitted/`.

| Run timestamp (UTC) | Status | Timed laps | Best lap (s) | Total (s) | Collisions |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261018T054321Z | COMPLETE | 10 | 16.9303 | 169.3975 | 0 |
| 20261018T063535Z | DISQUALIFIED | 0 | — | — | 11 |
| 20261018T063720Z | COMPLETE | 10 | 16.8897 | 168.9701 | 0 |

The best completed run achieved a 168.9701-second 10-lap total,
with a fastest lap of 16.8897 seconds and zero collisions.
Two of three submitted runs completed successfully.

The two completed runs reported real-time factors of 0.579 and
0.564 respectively.

### The machine that produced them
| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| AMD Ryzen 7 5800H with Radeon Graphics | NVIDIA GeForce RTX 3060 Laptop GPU; AMD Radeon Graphics | 13.9 GB (Windows-reported) | Windows 11 + Ubuntu WSL + Docker Desktop | 0.564–0.579x (completed runs) |

The simulator runs asynchronously through ROS. Closed-loop timing and machine load
may affect behavior. We observed performance differences between runs; a faster
three-lap braking experiment did not establish a better ten-lap submission.

## Anything precomputed
config/icra26_clearance.csv is the selected map-derived trajectory.
Its SHA256 is cad7650df64be515b3807887bdff386b2224b514637f405f06a6ffa695425ef6.
Existing generation code is in race_ws/src/team_driver/offline_racing/.
Reproduction commands (run from the repository root in the prepared environment):

```bash
python3 scripts/track_tool.py centerline --output race_ws/src/team_driver/config/generated_centreline.csv
python3 race_ws/src/team_driver/offline_racing/generate_clearance.py --map maps/icra26.yaml --centreline race_ws/src/team_driver/config/generated_centreline.csv --output race_ws/src/team_driver/config/clearance_seed.csv
python3 race_ws/src/team_driver/offline_racing/refine_clearance_time.py --map maps/icra26.yaml --centreline race_ws/src/team_driver/config/generated_centreline.csv --seed race_ws/src/team_driver/config/clearance_seed.csv --output race_ws/src/team_driver/config/icra26_clearance.csv
```
These are optional reproduction steps, not steps to run immediately before racing.
Optimisation-library differences can change the generated trajectory; the shipped
CSV is the artifact actually raced. No claim of global minimum-time optimality.

## Dependencies we added

None. Both `requirements.txt` and `apt-packages.txt` contain only
comments; no additional pip or apt packages are required.

The runtime uses ROS 2 Jazzy, `rclpy`, `ament_index_python`, and NumPy,
which are provided by the official judging environment.

The offline racing-line generation scripts additionally use SciPy,
Pillow and PyYAML, also available in the official environment.

## Third-party code and references
- Official NTU DeepSpeed icra26 map, centreline generation and simulator interfaces.
- TUMFTM/global_racetrajectory_optimization and BDEvan5/f1tenth_benchmarks as
  trajectory-planning references; the retained trajectory optimizer is original code.
- The inactive prediction_model.py fixture adapts f1tenth_gym single-track dynamics
  and constraints under MIT; retain LICENSE_F1TENTH_PREDICTION.txt. Prediction is off.

## AI assistance
ChatGPT assisted with source review, public-source research, controller/trajectory
code, test scripts, interpretation of uploaded results and packaging. The team ran
the official simulator locally and should be able to explain the submitted code.

## Known issues
The LiDAR guard and map-clearance checks are heuristic and do not guarantee safety.
The existing speed-command slew limit can temporarily keep a command above a newly
lowered guard cap. The local simulator-core screening did not reproduce every ROS
failure and was not treated as official reliability evidence. More time would be
used for actuator/timing identification, trajectory optimization and repeat races.

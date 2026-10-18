"""Candidate centreline pursuit + speed preview + reactive LiDAR guard.

Independent of ROS 2, so it can be unit-tested on a laptop.
Original implementation based on standard pure-pursuit and follow-the-gap ideas.
The official map centreline is shipped by NTU DeepSpeed; do not alter judging files.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .prediction_model import predict_pose
except ImportError:
    from prediction_model import predict_pose

WHEELBASE_M = 0.33
STEERING_LIMIT_RAD = 0.4189


def clamp(x, lower, upper):
    return float(np.clip(x, lower, upper))


def angle_wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


class Centreline:
    """Closed loop of x/y points listed in driving order."""
    def __init__(self, xy):
        p = np.asarray(xy, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] < 2 or p.shape[0] < 12:
            raise ValueError("Need >=12 centreline points with x,y columns")
        p = p[:, :2]
        if not np.all(np.isfinite(p)):
            raise ValueError("Centreline has NaN/inf values")
        if np.linalg.norm(p[0] - p[-1]) < 1e-5:
            p = p[:-1]
        self.xy = p
        self.n = len(p)
        delta = np.roll(p, -1, axis=0) - p
        self.segment_lengths = np.maximum(np.linalg.norm(delta, axis=1), 1e-5)
        self.headings = np.arctan2(delta[:, 1], delta[:, 0])
        # A 0.6-0.8 m chord on a 0.2 m-sampled centreline smooths small pixel jaggies.
        k = max(2, int(round(0.65 / float(np.median(self.segment_lengths)))))
        k = min(k, self.n // 8)
        a = np.roll(p, k, axis=0)
        b = p
        c = np.roll(p, -k, axis=0)
        ab = b - a
        bc = c - b
        ac = c - a
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
        self.curvature = 2.0 * cross / np.maximum(denom, 1e-6)

    def nearest(self, x, y, yaw, previous: Optional[int]):
        pos = np.array([x, y])
        if previous is None:
            indices = np.arange(self.n)
        else:
            # Select locally first, so two parallel passes of a hairpin do not swap.
            indices = (previous + np.arange(-17, 30)) % self.n
            nearby_d2 = np.sum((self.xy[indices] - pos) ** 2, axis=1)
            if float(np.min(nearby_d2)) > 1.8 ** 2:
                indices = np.arange(self.n)  # initial acquisition / large displacement
        d2 = np.sum((self.xy[indices] - pos) ** 2, axis=1)
        direction_penalty = 0.28 * (1 - np.cos(self.headings[indices] - yaw))
        return int(indices[np.argmin(d2 + direction_penalty)])

    def lookahead(self, start_index, distance):
        travelled = 0.0
        idx = int(start_index)
        for _ in range(self.n):
            step = float(self.segment_lengths[idx])
            nxt = (idx + 1) % self.n
            if travelled + step >= distance:
                f = (distance - travelled) / step
                return (1.0 - f) * self.xy[idx] + f * self.xy[nxt]
            travelled += step
            idx = nxt
        return self.xy[idx]

    def projected_lookahead(self, index, x, y, distance):
        """Project onto either segment adjoining the heading-selected waypoint.

        Restrict the projection locally to avoid switching to another hairpin arm.
        This affects the pursuit target only; speed preview and safety checks
        still use the original selected waypoint and error definition.
        """
        pos = np.array([x, y], dtype=np.float64)
        best = None
        for start in ((int(index) - 1) % self.n, int(index)):
            end = (start + 1) % self.n
            delta = self.xy[end] - self.xy[start]
            norm2 = float(delta @ delta)
            fraction = clamp(float((pos - self.xy[start]) @ delta) /
                             max(norm2, 1e-12), 0.0, 1.0)
            projection = self.xy[start] + fraction * delta
            error2 = float(np.sum((pos - projection) ** 2))
            if best is None or error2 < best[0]:
                best = (error2, start, fraction)
        _, start, fraction = best
        return self.lookahead(start, distance + fraction * self.segment_lengths[start])

    def corner_speed(self, index, max_speed, lateral_accel=2.5, braking=3.5,
                     preview_distance=7.0, min_speed=0.8):
        """Brake in advance of a bend using v^2 = u^2 + 2as."""
        best = max_speed
        distance = 0.0
        idx = int(index)
        for _ in range(self.n):
            k = abs(float(self.curvature[idx]))
            allowed_at_point = clamp(math.sqrt(lateral_accel / max(k, 1e-4)), min_speed, max_speed)
            allowed_now = math.sqrt(allowed_at_point ** 2 + 2.0 * braking * distance)
            best = min(best, allowed_now)
            distance += float(self.segment_lengths[idx])
            idx = (idx + 1) % self.n
            if distance > preview_distance:
                break
        return clamp(best, min_speed, max_speed)


def lidar_inflate(ranges, angles, radius=0.25, jump=0.4, cutoff=5.0):
    """Extend nearer LiDAR returns into far-side edges to account for car width."""
    scan = np.asarray(ranges, dtype=np.float64)
    adjusted = scan.copy()
    if len(scan) < 3:
        return adjusted
    increment = abs(float(angles[1] - angles[0]))
    for i in np.flatnonzero((np.abs(np.diff(scan)) > jump) &
                            (np.minimum(scan[:-1], scan[1:]) < cutoff)):
        near = min(float(scan[i]), float(scan[i+1]))
        if near <= 0.0:
            continue
        count = min(65, int(math.ceil(math.atan2(radius, near) / increment)))
        if scan[i] < scan[i+1]:  # close surface on left; extend towards right (increasing indices)
            adjusted[i+1:min(len(scan), i+count+1)] = np.minimum(
                adjusted[i+1:min(len(scan), i+count+1)], near)
        else:
            adjusted[max(0, i-count+1):i+1] = np.minimum(
                adjusted[max(0, i-count+1):i+1], near)
    return adjusted


def sector_min(scan, angles, heading, half_width=0.10):
    mask = np.abs(angles - heading) < half_width
    return float(np.min(scan[mask])) if np.any(mask) else float(np.min(scan))


def find_safe_gap(adjusted, angles, desired_angle, max_abs_angle=1.00,
                  minimum_range=0.85):
    """Pick centre of a navigable gap, biased to direction of the planned path."""
    good = ((np.abs(angles) < max_abs_angle) & (adjusted >= minimum_range))
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    candidates = []
    for s, e in zip(starts, ends):
        if e - s < 10:  # a gap narrower than ~3 degrees is not meaningful
            continue
        # Aiming at gap centre keeps both corners far from the car.
        mid = (s + e - 1) // 2
        proposed = float(angles[mid])
        # Prefer path direction without selecting a slit between nearby cones.
        gap_width = float(angles[e-1] - angles[s])
        score = (1.8 * abs(angle_wrap(proposed - desired_angle))
                 + 0.3 * abs(proposed) - 0.22 * gap_width)
        candidates.append((score, proposed))
    return min(candidates)[1] if candidates else None


@dataclass
class ControlCommand:
    steering: float
    speed: float
    mode: str
    nearest_index: int
    path_error: float
    front_range: float
    planned_speed: float = 0.0
    guarded_speed: float = 0.0


class CyclicSpeedProfile:
    """Squared-speed relaxation on every edge of a closed trajectory.

    Point-mass longitudinal constraints, independent lateral speed caps.
    Not a tire/friction-circle model or a collision guarantee.
    """
    def __init__(self, line, max_speed, lateral_accel, acceleration, braking):
        values = np.array([max_speed, lateral_accel, acceleration, braking])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Profile limits must be finite and positive")
        self.line = line
        self.braking = float(braking)
        cap2 = np.minimum(max_speed ** 2,
                          lateral_accel / np.maximum(np.abs(line.curvature), 1e-4))
        q = cap2.copy()
        for _ in range(line.n + 1):
            before = q.copy()
            for i in range(line.n - 1, -1, -1):
                j = (i + 1) % line.n
                q[i] = min(q[i], q[j] + 2.0 * braking * line.segment_lengths[i])
            for i in range(line.n):
                j = (i + 1) % line.n
                q[j] = min(q[j], q[i] + 2.0 * acceleration * line.segment_lengths[i])
            if np.max(np.abs(before - q)) < 1e-10:
                break
        else:
            raise RuntimeError("Cyclic velocity profile did not converge")
        self.speed = np.sqrt(q)
        self.cap = np.sqrt(cap2)
        self.ideal_time = float(np.sum(2.0 * line.segment_lengths /
                                      (self.speed + np.roll(self.speed, -1))))

    def command(self, index, actual_speed, response_delay):
        """Reserve travel during delay before applying the braking envelope.

        Preview can only reduce the local profile speed; it does not request
        higher corner-exit speed before the vehicle reaches the exit.
        """
        delay_distance = max(0.0, float(actual_speed)) * response_delay
        best = float(self.speed[index])
        distance = 0.0
        i = int(index)
        horizon = best * best / (2.0 * self.braking) + delay_distance
        for _ in range(self.line.n):
            available = max(0.0, distance - delay_distance)
            best = min(best, math.sqrt(self.speed[i] ** 2 +
                                       2.0 * self.braking * available))
            distance += float(self.line.segment_lengths[i])
            i = (i + 1) % self.line.n
            if distance > horizon:
                break
        return best


class HybridController:
    def __init__(self, centreline, max_speed=3.3, lateral_accel=2.5,
                 brake_accel=3.5, accel_limit=5.0, lookahead_scale=0.90,
                 project_progress=False, use_speed_profile=False,
                 response_delay=0.0, prediction_horizon=0.0):
        self.line = Centreline(centreline)
        self.max_speed = float(max_speed)
        self.lateral_accel = float(lateral_accel)
        self.brake_accel = float(brake_accel)
        self.accel_limit = float(accel_limit)
        self.lookahead_scale = float(lookahead_scale)
        self.project_progress = bool(project_progress)
        if not (0.5 <= self.lookahead_scale <= 1.5) or self.accel_limit <= 0:
            raise ValueError("Invalid lookahead scale or acceleration limit")
        self.use_speed_profile = bool(use_speed_profile)
        self.response_delay = float(response_delay)
        if not math.isfinite(self.response_delay) or not 0 <= self.response_delay <= 0.5:
            raise ValueError("response_delay must be between 0 and 0.5 seconds")
        self.profile = (CyclicSpeedProfile(self.line, self.max_speed,
                        self.lateral_accel, self.accel_limit, self.brake_accel)
                        if self.use_speed_profile else None)
        self.last_index = None
        self.previous_speed = None
        self.previous_steer = None
        self.prediction_horizon = float(prediction_horizon)
        if not math.isfinite(self.prediction_horizon) or not 0 <= self.prediction_horizon <= .1:
            raise ValueError("prediction_horizon must be between 0 and 0.1 seconds")
        self.motion_state = None  # yaw rate and CoG slip angle from bridge odometry
        self.estimated_steer = 0.0
        self.prediction_delta = 0.0


    def step(self, x, y, yaw, actual_speed, ranges, angles, dt=0.025):
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        if ranges.shape != angles.shape or len(ranges) < 50:
            raise ValueError("LiDAR data shape is invalid")
        if not np.all(np.isfinite(ranges)):
            raise ValueError("Call preprocessing first to clean LiDAR scan")
        idx = self.line.nearest(x, y, yaw, self.last_index)
        self.last_index = idx
        path_error = float(np.hypot(x - self.line.xy[idx, 0],
                                    y - self.line.xy[idx, 1]))
        lookahead_m = self.lookahead_scale * clamp(0.74 + 0.19 * actual_speed, 0.75, 1.65)
        target = (self.line.projected_lookahead(idx, x, y, lookahead_m)
                  if self.project_progress else self.line.lookahead(idx, lookahead_m))
        dx, dy = float(target[0] - x), float(target[1] - y)
        target_bearing = angle_wrap(math.atan2(dy, dx) - yaw)
        dist_to_target = max(0.2, math.hypot(dx, dy))
        curvature = 2.0 * math.sin(target_bearing) / dist_to_target
        steering = math.atan(WHEELBASE_M * curvature)
        self.prediction_delta = 0.0
        if (self.prediction_horizon > 0 and self.motion_state is not None
                and self.previous_steer is not None and actual_speed >= 1.0
                and .005 <= dt <= .05 and path_error <= .45):
            yaw_rate, beta = self.motion_state
            if math.isfinite(yaw_rate) and math.isfinite(beta) and abs(yaw_rate) < 10 and abs(beta) < .5:
                try:
                    px, py, pyaw = predict_pose(x, y, yaw, actual_speed,
                        yaw_rate, beta, self.estimated_steer, self.previous_steer,
                        self.previous_speed, self.prediction_horizon)
                    if all(math.isfinite(v) for v in (px, py, pyaw)):
                        pidx = self.line.nearest(px, py, pyaw, idx)
                        pt = self.line.lookahead(pidx, lookahead_m)
                        pdx, pdy = float(pt[0]-px), float(pt[1]-py)
                        predicted = math.atan(WHEELBASE_M * 2.0 * math.sin(
                            angle_wrap(math.atan2(pdy, pdx)-pyaw)) / max(.2, math.hypot(pdx,pdy)))
                        self.prediction_delta = clamp(predicted-steering, -.04, .04)
                        steering += self.prediction_delta
                except (ValueError, OverflowError, FloatingPointError):
                    pass  # retain original Pure Pursuit on invalid prediction
        if self.profile is not None:
            target_speed = self.profile.command(idx, actual_speed, self.response_delay)
        else:
            target_speed = self.line.corner_speed(
                idx, self.max_speed, self.lateral_accel, self.brake_accel)
        planned_speed = target_speed
        mode = "PURE_PURSUIT"

        # LiDAR guard: the laser is 0.275 m ahead of the chassis origin.
        expanded = lidar_inflate(ranges, angles)
        forward = sector_min(expanded, angles, 0.0, 0.12)
        on_course = sector_min(expanded, angles,
                               clamp(target_bearing, -0.8, 0.8), 0.11)

        # Use caution if the car has drifted off the centreline.
        if path_error > 0.6:
            target_speed = min(target_speed, 1.2)

        # Turn-aware LiDAR speed guard: a wall straight ahead may be irrelevant
        # when following a sharp bend. Check the inflated LiDAR sector pointed
        # along the planned pursuit bearing, leaving a buffer for braking.
        # Keep the hard emergency guard and gap recovery below unchanged.
        if forward < 1.2:
            if (abs(target_bearing) >= 0.40 and path_error <= 0.28
                    and on_course >= 1.0):
                path_stopping_distance = max(0.0, on_course - 0.55)
                path_speed_cap = math.sqrt(2.0 * self.brake_accel * path_stopping_distance)
                target_speed = min(target_speed, max(0.6, path_speed_cap))
                mode = "PATH_AWARE_BEND"
            else:
                target_speed = min(target_speed, max(0.6, 1.35 * (forward - 0.15)))
                mode = "FRONT_LIMIT"

        # Only deviate from the reference if the intended direction is obstructed.
        # This switch is heuristic: tune with the official collision logs.
        trigger = max(0.68, 0.30 + 0.25 * max(actual_speed, 0.0))
        if on_course < trigger and path_error < 1.2:
            safe_angle = find_safe_gap(expanded, angles, target_bearing,
                                       minimum_range=max(0.75, trigger))
            # Never follow a large side gap away from the known legal route:
            # gaps between a row of cones are not legal shortcuts on icra26.
            if (safe_angle is not None and
                    abs(angle_wrap(safe_angle - target_bearing)) <= 0.38):
                steering = math.atan(2.0 * WHEELBASE_M * math.sin(safe_angle) / 1.0)
                target_speed = min(target_speed, 1.15)
                mode = "GAP_RECOVERY"
            else:
                target_speed = min(target_speed, 0.55)
                mode = "PATH_GUARD"

        if forward < 0.38:
            target_speed = min(target_speed, 0.45)
            mode = "CAUTION" if mode == "PURE_PURSUIT" else mode

        guarded_speed = target_speed
        dt = clamp(dt, 0.005, 0.10)
        steering = clamp(steering, -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD)
        if self.previous_steer is not None:
            step = 3.2 * dt  # simulator max steering-rate limit
            steering = clamp(steering,
                             self.previous_steer - step,
                             self.previous_steer + step)
        if self.prediction_horizon > 0:
            self.estimated_steer += clamp((10*3.2/.8378)*(steering-self.estimated_steer), -3.2, 3.2)*dt
            self.estimated_steer = clamp(self.estimated_steer, -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD)
        self.previous_steer = steering
        if self.previous_speed is None:
            self.previous_speed = max(0.65, min(1.0, target_speed))
        else:
            upper = self.previous_speed + self.accel_limit * dt
            lower = self.previous_speed - 5.0 * dt
            self.previous_speed = clamp(target_speed, lower, upper)
        return ControlCommand(steering, self.previous_speed,
                              mode, idx, path_error, forward,
                              planned_speed, guarded_speed)

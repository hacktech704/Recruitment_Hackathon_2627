"""F1TENTH single-track prediction. See LICENSE_F1TENTH.txt for MIT attribution.
Adapted from installed f1tenth_gym: removed JIT; added rear-axle pose predictor.
"""
import numpy as np


def upper_accel_limit(vel, a_max, v_switch):
    """
    Upper acceleration limit, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_max (float): maximum allowed acceleration, symmetrical
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)

        Returns:
            positive_accel_limit (float): adjusted acceleration
    """
    if vel > v_switch:
        pos_limit = a_max * (v_switch / vel)
    else:
        pos_limit = a_max

    return pos_limit

def accl_constraints(vel, a_long_d, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            a_long_d (float): unconstrained desired acceleration in the direction of travel.
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration, symmetrical
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    uac = upper_accel_limit(vel, a_max, v_switch)

    if (vel <= v_min and a_long_d <= 0) or (vel >= v_max and a_long_d >= 0):
        a_long = 0.0
    elif a_long_d <= -a_max:
        a_long = -a_max
    elif a_long_d >= uac:
        a_long = uac
    else:
        a_long = a_long_d

    return a_long


def steering_constraint(
    steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max
):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (
        steering_angle >= s_max and steering_velocity >= 0
    ):
        steering_velocity = 0.0
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity

def pid_steer(steer, current_steer, max_sv, kp):
    """Steering-angle tracking: a saturated P controller.

    ``sv = clip(kp * (steer - current_steer), -max_sv, +max_sv)``.

    Args:
        steer: Target steering angle in radians.
        current_steer: Current steering angle in radians.
        max_sv: Steering-velocity saturation in rad/s.
        kp: Proportional gain. ``kp <= 0`` selects the legacy bang-bang relay
            (full ``±max_sv`` whenever the error exceeds 1e-4 rad), kept as an
            explicit escape hatch — the relay's minimum increment per step
            (``max_sv * dt``) dwarfs its deadband, so it limit-cycles forever.

    Returns:
        The commanded steering velocity in rad/s.
    """
    steer_diff = steer - current_steer
    if kp <= 0.0:
        if np.fabs(steer_diff) > 1e-4:
            return (steer_diff / np.fabs(steer_diff)) * max_sv
        return 0.0
    sv = kp * steer_diff
    if sv > max_sv:
        sv = max_sv
    elif sv < -max_sv:
        sv = -max_sv
    return sv


def pid_accl(speed, current_speed, max_a, max_v, min_v):
    """Speed tracking: a P controller with four gain quadrants.

    The gain depends on the sign of the current speed and of the speed error. The
    quadrant test is ``current_speed > 0.0``, so a car at exactly rest uses the
    weaker reverse gains.

    Args:
        speed: Target speed in m/s.
        current_speed: Current speed in m/s.
        max_a: Acceleration limit in m/s^2.
        max_v: Forward speed limit in m/s.
        min_v: Reverse speed limit in m/s (negative).

    Returns:
        The commanded acceleration in m/s^2 (before ``accl_constraints``).
    """
    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.0:
        if vel_diff > 0:
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if vel_diff > 0:
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl

import numpy as np




def vehicle_dynamics_st(x: np.ndarray, u_init: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single Track Vehicle Dynamics.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x0, x1, x2, x3, x4, x5, x6)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration
            params (np.ndarray (88, )): flat float32 parameter array in VehicleParameters declaration order; this model reads indices 0-17 (a dict raises a numba typing error):
                mu (float): friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel spin
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # States
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    PSI_DOT = x[5]
    BETA = x[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = np.arctan2(np.sin(BETA), np.cos(BETA))

    # gravity constant m/s^2
    g = 9.81
    mu = params[0]
    C_Sf = params[1]  # front tire cornering stiffness [N/rad]  CF
    C_Sr = params[2]  # rear tire cornering stiffness [N/rad]  CR
    lf = params[3]  # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params[4]  # distance from spring mass center of gravity to rear axle [m]  LENB
    h = params[5]  # M_s center of gravity above ground [m]  HS
    m = params[6]  # vehicle mass [kg]  MASS
    I = params[7]  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    s_min = params[8]  # minimum steering angle [rad]
    s_max = params[9]  # maximum steering angle [rad]
    sv_min = params[10]  # minimum steering velocity [rad/s]
    sv_max = params[11]  # maximum steering velocity [rad/s]
    v_switch = params[12]  # switching velocity [m/s]
    a_max = params[13]  # maximum absolute acceleration [m/s^2]
    v_min = params[14]  # minimum velocity [m/s]
    v_max = params[15]  # maximum velocity [m/s]
    # width = params[16]  # vehicle width [m]
    # length = params[17]  # vehicle length [m]

    # constraints
    u = np.array(
        [
            steering_constraint(
                DELTA,
                u_init[0],
                s_min,
                s_max,
                sv_min,
                sv_max,
            ),
            accl_constraints(
                V,
                u_init[1],
                v_switch,
                a_max,
                v_min,
                v_max,
            ),
        ]
    )
    # Controls
    STEER_VEL = u[0]
    ACCL = u[1]

    # switch to kinematic model for small velocities
    if V < 0.5:
        # wheelbase
        lwb = lf + lr
        BETA_HAT = np.arctan(np.tan(DELTA) * lr / lwb)
        BETA_DOT = (
            (1 / (1 + (np.tan(DELTA) * (lr / lwb)) ** 2))
            * (lr / (lwb * np.cos(DELTA) ** 2))
            * STEER_VEL
        )
        f = np.array(
            [
                V * np.cos(PSI + BETA_HAT),  # X_DOT
                V * np.sin(PSI + BETA_HAT),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                V * np.cos(BETA_HAT) * np.tan(DELTA) / lwb,  # PSI_DOT
                (1 / lwb)
                * (
                    ACCL * np.cos(BETA) * np.tan(DELTA)
                    - V * np.sin(BETA) * np.tan(DELTA) * BETA_DOT
                    + ((V * np.cos(BETA) * STEER_VEL) / (np.cos(DELTA) ** 2))
                ),  # PSI_DOT_DOT
                BETA_DOT,  # BETA_DOT
            ]
        )
    else:
        # system dynamics
        glr = g * lr - ACCL * h
        glf = g * lf + ACCL * h
        f = np.array(
            [
                V * np.cos(PSI + BETA),  # X_DOT
                V * np.sin(PSI + BETA),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                PSI_DOT,  # PSI_DOT
                (
                    (mu * m)
                    / (I * (lf + lr))
                )
                * (
                    lf * C_Sf * (glr) * DELTA
                    + (
                        lr * C_Sr * (glf)
                        - lf * C_Sf * (glr)
                    )
                    * BETA
                    - (
                        lf * lf * C_Sf * (glr)
                        + lr * lr * C_Sr * (glf)
                    )
                    * (PSI_DOT / V)
                ),  # PSI_DOT_DOT
                (mu / (V * (lr + lf)))
                * (
                    C_Sf * (glr) * DELTA
                    - (C_Sr * (glf) + C_Sf * (glr)) * BETA
                    + (
                        C_Sr * (glf) * lr
                        - C_Sf * (glr) * lf
                    )
                    * (PSI_DOT / V)
                )
                - PSI_DOT,  # BETA_DOT
            ]
        )

    return f


"""Short-horizon dynamic pose prediction, not MPC or a collision guarantee."""
import math
import numpy as np


PARAMS=np.array([1.0489,4.718,5.4562,.15875,.17145,.074,3.74,.04712,-.4189,.4189,-3.2,3.2,7.319,9.51,-5.,20.,.31,.58])
def predict_pose(x,y,yaw,v,yaw_rate,beta,steer,target_steer,target_speed,horizon):
    state=np.array([x+.17145*math.cos(yaw),y+.17145*math.sin(yaw),steer,v,yaw,yaw_rate,beta])
    n=max(1,math.ceil(horizon/.01));dt=horizon/n
    for _ in range(n):
        u=np.array([pid_steer(target_steer,state[2],3.2,10*3.2/.8378),pid_accl(target_speed,state[3],9.51,20.,-5.)])
        a=vehicle_dynamics_st(state,u,PARAMS);b=vehicle_dynamics_st(state+.5*dt*a,u,PARAMS)
        c=vehicle_dynamics_st(state+.5*dt*b,u,PARAMS);d=vehicle_dynamics_st(state+dt*c,u,PARAMS)
        state+=dt*(a+2*b+2*c+d)/6
    return state[0]-.17145*math.cos(state[4]),state[1]-.17145*math.sin(state[4]),state[4]

#!/usr/bin/env python3

import signal
import numpy as np
import rclpy

from crazyflie_py import Crazyswarm


DRONE_NAME = "cf_7"

TAKEOFF_HEIGHT = 1.50
TAKEOFF_DURATION = 3.0

PRE_TRAJECTORY_HOVER_TIME = 10.0
RECOVERY_TIME = 5

LAND_HEIGHT = 0.04
LAND_DURATION = 3.0

CHECK_START_POSITION = True
MAX_START_POSITION_ERROR = 0.25

MIN_SAFE_Z = 0.40
MAX_SAFE_Z = 2.60
MAX_HORIZONTAL_DISPLACEMENT_FROM_START = 1.50

NOMINAL_SAMPLE_RATE = 30.0
TIMING_SLEEP = 0.001


# (
#   time_s,
#   [px, py, pz],
#   [vx, vy, vz],
#   [ax, ay, az],
#   yaw_rad,
#   yaw_rate_rad_s,
# )
PVA_SEQUENCE = [
    (0.0,
     [0.0, 0.0, 1.502255342821949],
     [0.0, 0.0, -0.00015739809255954976],
     [13.003387008252812, 0.021062814090227602, 4.741460006405412],
     0.0, 0.0),

    (0.03333333333333333,
     [0.005898221856602676, 9.553907020508786e-06, 1.504396316519195],
     [0.350529139692009, 0.0005677851545796784, 0.12709620719208903],
     [11.141696973789175, 0.018047258907208298, 4.015305677055965],
     0.0, 0.0),

    (0.06666666666666667,
     [0.0232283916601369, 3.779601992894694e-05, 1.5106538919073602],
     [0.6909268096869977, 0.0011392684386130594, 0.24871568517840317],
     [9.29027706947697, 0.016115104683462014, 3.2917884686849614],
     0.0, 0.0),

    (0.1,
     [0.05180908363395075, 8.53983830635262e-05, 1.5209160744358372],
     [1.020410069687884, 0.001710800295405633, 0.36573465374388725],
     [8.9255179258721, 0.024521503981817627, 4.0127271698878815],
     0.0, 0.0),

    (0.13333333333333333,
     [0.08986004219058205, 0.0001534806360614125, 1.5349141958984422],
     [1.2601084920375385, 0.0023704958851325383, 0.47324076825829653],
     [7.612296374205814, 0.02510098745666928, 3.4402236143950917],
     0.0, 0.0),

    (0.16666666666666666,
     [0.13571142013018517, 0.00024354200926182307, 1.5524293495506218],
     [1.492003297716344, 0.0030570882316106993, 0.5783915792909652],
     [6.319056124855869, 0.019945229349290633, 2.866561143462538],
     0.0, 0.0),

    (0.2,
     [0.18922531189557606, 0.00035737257956843925, 1.5734386213936753],
     [1.7165118355860107, 0.003765292747654156, 0.6812371076138047],
     [2.5200407510674268, 0.03082791718086364, 3.6131310987955203],
     0.0, 0.0),

    (0.23333333333333334,
     [0.2475756778503165, 0.0004966421753357768, 1.5977720391211163],
     [1.7837518771634764, 0.004650415413136689, 0.7780938390348449],
     [2.149501301031415, 0.04161000507723811, 3.1162640306711875],
     0.0, 0.0),

    (0.26666666666666666,
     [0.3081194284492589, 0.0006628052653643435, 1.6252865041711047],
     [1.8489664917365587, 0.005353833499191576, 0.8735693459166856],
     [1.7641154247473436, 0.02684191298586833, 2.614344776528921],
     0.0, 0.0),

    (0.3,
     [0.3708125932922788, 0.0008580475032615954, 1.6559871433171665],
     [1.9120338078311008, 0.006313218874618358, 0.9677200731444893],
     [-4.819528971361933, 0.030984095775842935, 3.313341991050379],
     0.0, 0.0),

    (0.3333333333333333,
     [0.43235324899808636, 0.0010818714763844554, 1.6897315790837244],
     [1.7816449492100066, 0.0071408452427981365, 1.056087031988507],
     [-4.139462933645097, 0.03018793704480794, 2.824041404295695],
     0.0, 0.0),

    (0.36666666666666664,
     [0.4896287402405609, 0.0013333847015062573, 1.7263584594527293],
     [1.653697869461361, 0.007874105486544764, 1.1420184957997392],
     [-3.5300510203550517, 0.01488650802496758, 2.3351131211365708],
     0.0, 0.0),

    (0.4,
     [0.542648043364236, 0.0016063818046130256, 1.765838218466043],
     [1.5288098673427999, 0.00857298302972825, 1.2261086988441912],
     [-9.806709236966867, 0.03173404244863054, 2.244934815767465],
     0.0, 0.0),

    (0.43333333333333335,
     [0.5891554334276029, 0.001908100401689001, 1.8077079736268329],
     [1.2641845568993222, 0.00952609021253444, 1.2853950738894213],
     [-8.408695674792307, 0.028251752264201537, 1.8852412772142022],
     0.0, 0.0),

    (0.4666666666666667,
     [0.627019669780565, 0.002237457992123858, 1.851496011852951],
     [1.005808749074592, 0.010152848292691917, 1.3420550157858053],
     [-7.092196152963002, 0.011613526115042749, 1.5167094143875022],
     0.0, 0.0),

    (0.5,
     [0.656307451144411, 0.0025815060853577264, 1.8971455932848065],
     [0.7540048444956416, 0.01044273018446107, 1.3963054149759107],
     [-10.785426122413185, 0.0023660138490324556, -1.2214502764888948],
     0.0, 0.0),

    (0.5333333333333333,
     [0.6765495796012865, 0.002929360066820506, 1.9430997474329545],
     [0.4633750326367292, 0.010414709325831275, 1.3607894720942266],
     [-9.233906862413747, -0.00011846983510363599, -1.1537902581787844],
     0.0, 0.0),

    (0.5666666666666667,
     [0.6873097858694005, 0.0032780248032569007, 1.9878440644901603],
     [0.18042779307538587, 0.01055898405921645, 1.3231419272699951],
     [-7.742589423385173, 0.010534471538827205, -1.0976714838294248],
     0.0, 0.0),

    (0.6,
     [0.6886954911617102, 0.0036401422779084843, 2.0312929407420515],
     [-0.09436940854695049, 0.011263039872474945, 1.2837386249076819],
     [-11.527857078339068, 0.020408993930526087, -4.635719301917902],
     0.0, 0.0),

    (0.6333333333333333,
     [0.6803402054772155, 0.004024416479806804, 2.071924801114928],
     [-0.4036099094559753, 0.011750064656021948, 1.1545706423912128],
     [-9.806807150278987, 0.008311358893810768, -4.151615817396206],
     0.0, 0.0),
]


abort_requested = False
emergency_sent = False


def sigint_handler(signum, frame):
    global abort_requested
    if not abort_requested:
        print()
        print("CTRL+C -> emergency requested")
    abort_requested = True


def emergency(cf, time_helper):
    global emergency_sent

    if emergency_sent:
        return

    emergency_sent = True

    print()
    print("================================")
    print("!!! EMERGENCY STOP !!!")
    print("================================")

    cf.emergency()

    end_time = time_helper.time() + 0.5
    while rclpy.ok() and time_helper.time() < end_time:
        rclpy.spin_once(time_helper.node, timeout_sec=0.01)


def validate_sequence():
    if not PVA_SEQUENCE:
        raise RuntimeError("PVA_SEQUENCE cannot be empty.")

    previous_time = None

    for index, sample in enumerate(PVA_SEQUENCE):
        if len(sample) != 6:
            raise RuntimeError(
                f"Sample {index}: expected "
                "(time, position, velocity, acceleration, yaw, yaw_rate)."
            )

        time_s, position, velocity, acceleration, yaw, yaw_rate = sample

        if time_s < 0.0:
            raise RuntimeError(f"Sample {index}: time must be >= 0.")

        if previous_time is not None and time_s <= previous_time:
            raise RuntimeError(
                f"Sample {index}: timestamps must be strictly increasing."
            )

        previous_time = time_s

        for name, vector in (
            ("position", position),
            ("velocity", velocity),
            ("acceleration", acceleration),
        ):
            vector = np.asarray(vector, dtype=float)

            if vector.shape != (3,):
                raise RuntimeError(
                    f"Sample {index}: {name} must contain 3 values."
                )

            if not np.all(np.isfinite(vector)):
                raise RuntimeError(
                    f"Sample {index}: {name} contains non-finite values."
                )

        if not np.isfinite(yaw) or not np.isfinite(yaw_rate):
            raise RuntimeError(
                f"Sample {index}: yaw/yaw_rate contains non-finite values."
            )

        z = float(position[2])
        if not (MIN_SAFE_Z <= z <= MAX_SAFE_Z):
            raise RuntimeError(
                f"Sample {index}: desired z={z:.3f} m outside safety limits."
            )


def safety_violation(cf, start_position):
    position = np.asarray(cf.get_position(), dtype=float)

    if not np.all(np.isfinite(position)):
        return "position estimate is not finite"

    if position[2] < MIN_SAFE_Z:
        return (
            f"z={position[2]:.3f} m "
            f"< MIN_SAFE_Z={MIN_SAFE_Z:.3f} m"
        )

    if position[2] > MAX_SAFE_Z:
        return (
            f"z={position[2]:.3f} m "
            f"> MAX_SAFE_Z={MAX_SAFE_Z:.3f} m"
        )

    horizontal_displacement = np.linalg.norm(
        position[:2] - start_position[:2]
    )

    if horizontal_displacement > MAX_HORIZONTAL_DISPLACEMENT_FROM_START:
        return (
            f"horizontal displacement={horizontal_displacement:.3f} m "
            f"> limit={MAX_HORIZONTAL_DISPLACEMENT_FROM_START:.3f} m"
        )

    return None


def send_full_state_sample(
    cf,
    position,
    velocity,
    acceleration,
    yaw,
    yaw_rate,
):
    cf.cmdFullState(
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(acceleration, dtype=float),
        float(yaw),
        np.array([0.0, 0.0, float(yaw_rate)], dtype=float),
    )


def stream_hover(
    cf,
    time_helper,
    position,
    yaw,
    duration,
):
    zero = np.zeros(3, dtype=float)
    start_time = time_helper.time()

    while time_helper.time() - start_time < duration:
        if abort_requested:
            emergency(cf, time_helper)
            return False

        cf.cmdFullState(
            np.asarray(position, dtype=float),
            zero,
            zero,
            float(yaw),
            zero,
        )

        time_helper.sleepForRate(NOMINAL_SAMPLE_RATE)

    return True


def execute_pva_sequence(
    cf,
    time_helper,
    start_position,
):
    print()
    print("================================")
    print("DIRECT PVA SEQUENCE")
    print("================================")
    print(f"Samples: {len(PVA_SEQUENCE)}")
    print(f"Duration: {PVA_SEQUENCE[-1][0]:.4f} s")
    print("P, V and A are sent directly. No interpolation.")

    sequence_start = time_helper.time()
    max_lateness = 0.0

    for index, sample in enumerate(PVA_SEQUENCE):
        (
            time_s,
            position,
            velocity,
            acceleration,
            yaw,
            yaw_rate,
        ) = sample

        target_time = sequence_start + time_s

        while True:
            if abort_requested:
                emergency(cf, time_helper)
                return False

            remaining = target_time - time_helper.time()
            if remaining <= 0.0:
                break

            time_helper.sleep(min(TIMING_SLEEP, remaining))

        violation = safety_violation(cf, start_position)

        if violation is not None:
            print()
            print("SAFETY LIMIT -> " + violation)
            return False

        actual_time = time_helper.time() - sequence_start
        lateness = max(0.0, actual_time - time_s)
        max_lateness = max(max_lateness, lateness)

        send_full_state_sample(
            cf,
            position,
            velocity,
            acceleration,
            yaw,
            yaw_rate,
        )

        print(
            f"[{index + 1:02d}/{len(PVA_SEQUENCE):02d}] "
            f"t={time_s:.4f}s "
            f"p=({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}) "
            f"v=({velocity[0]:+.3f}, {velocity[1]:+.3f}, {velocity[2]:+.3f}) "
            f"a=({acceleration[0]:+.3f}, {acceleration[1]:+.3f}, {acceleration[2]:+.3f})"
        )

    time_helper.sleep(1.0 / NOMINAL_SAMPLE_RATE)

    print()
    print(f"Maximum send lateness: {1000.0 * max_lateness:.2f} ms")

    return True


def main():
    validate_sequence()

    swarm = Crazyswarm()
    allcfs = swarm.allcfs
    time_helper = swarm.timeHelper

    signal.signal(signal.SIGINT, sigint_handler)

    if DRONE_NAME not in allcfs.crazyfliesByName:
        raise RuntimeError(f"{DRONE_NAME} not found.")

    cf = allcfs.crazyfliesByName[DRONE_NAME]

    print()
    print("================================")
    print("Direct full-state PVA test")
    print("================================")
    print()
    print(f"Drone: {DRONE_NAME}")
    print(f"Nominal sample rate: {NOMINAL_SAMPLE_RATE:.1f} Hz")

    first_position = np.asarray(PVA_SEQUENCE[0][1], dtype=float)
    first_yaw = float(PVA_SEQUENCE[0][4])

    flight_active = False

    try:
        print()
        print("Arming...")
        cf.arm(True)
        flight_active = True
        time_helper.sleep(1.0)

        print(f"Taking off to {TAKEOFF_HEIGHT:.2f} m...")
        cf.takeoff(
            targetHeight=TAKEOFF_HEIGHT,
            duration=TAKEOFF_DURATION,
        )

        time_helper.sleep(TAKEOFF_DURATION + 1.0)

        if abort_requested:
            emergency(cf, time_helper)
            return

        actual_start_position = np.asarray(
            cf.get_position(),
            dtype=float,
        )

        start_error = np.linalg.norm(
            actual_start_position - first_position
        )

        print()
        print(f"Post-takeoff position: {actual_start_position}")
        print(f"First desired position: {first_position}")
        print(f"Start-position error: {start_error:.3f} m")

        if CHECK_START_POSITION and start_error > MAX_START_POSITION_ERROR:
            raise RuntimeError(
                "Post-takeoff position is too far from first PVA sample: "
                f"{start_error:.3f} m > {MAX_START_POSITION_ERROR:.3f} m"
            )

        print()
        print("Holding first PVA position before playback...")

        if not stream_hover(
            cf,
            time_helper,
            first_position,
            first_yaw,
            PRE_TRAJECTORY_HOVER_TIME,
        ):
            return

        completed = execute_pva_sequence(
            cf,
            time_helper,
            first_position,
        )

        if abort_requested:
            return

        recovery_position = np.asarray(
            cf.get_position(),
            dtype=float,
        )

        recovery_yaw = float(PVA_SEQUENCE[-1][4])

        print()
        print(f"Recovery position: {recovery_position}")
        print(f"Recovery time: {RECOVERY_TIME:.2f} s")

        if not stream_hover(
            cf,
            time_helper,
            recovery_position,
            recovery_yaw,
            RECOVERY_TIME,
        ):
            return

        print()
        print("Switching to high-level landing...")

        cf.notifySetpointsStop(
            remainValidMillisecs=200
        )

        cf.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION,
        )

        time_helper.sleep(LAND_DURATION + 1.0)

        cf.arm(False)
        flight_active = False

        print()
        print("Experiment complete.")

        if not completed:
            print("Trajectory stopped by runtime safety bound.")

    except Exception:
        if flight_active:
            emergency(cf, time_helper)
        raise


if __name__ == "__main__":
    main()

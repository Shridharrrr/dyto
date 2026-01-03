import yaml

from perception.camera import capture_image
from perception.detector import VehicleDetector
from perception.siren import detect_siren

from logic.pressure import calculate_pressure
from logic.signal_engine import allocate_green_times, override_signal
from logic.emergency import confirm_emergency

# Load configs
lanes_cfg = yaml.safe_load(open("config/lanes.yaml"))
weights_cfg = yaml.safe_load(open("config/weights.yaml"))
emergency_cfg = yaml.safe_load(open("config/emergency.yaml"))

weights = weights_cfg["vehicle_weights"]
lanes = [lane["id"] for lane in lanes_cfg["lanes"]]

detector = VehicleDetector("models/yolov8n.pt")

def run_signal_cycle():

    pressures = {}
    emergency_triggered = False
    emergency_lane = None

    print("\n🚦 Signal Cycle Start\n")

    # STEP 1: Check siren first (interrupt-based)
    for lane in lanes_cfg["lanes"]:
        lane_id = lane["id"]

        siren_audio = f"data/audio/{lane_id}_siren.wav"
        siren_detected, siren_conf = detect_siren(siren_audio)

        if siren_detected:
            print(f"🚨 Siren detected on Lane {lane_id} (conf={siren_conf})")

            image = capture_image(lane["camera_source"])
            counts = detector.detect(image)

            if confirm_emergency(
                counts,
                siren_conf,
                emergency_cfg["override"]["min_confidence"]
            ):
                emergency_triggered = True
                emergency_lane = lane_id
                break

    # STEP 2: Emergency override
    if emergency_triggered:
        print(f"\n🛑 EMERGENCY OVERRIDE: Lane {emergency_lane}\n")
        return override_signal(
            emergency_lane,
            lanes,
            emergency_cfg["override"]["override_green_time"]
        )

    # STEP 3: Normal DyTo logic
    for lane in lanes_cfg["lanes"]:
        image = capture_image(lane["camera_source"])
        counts = detector.detect(image)
        pressure = calculate_pressure(counts, weights)
        pressures[lane["id"]] = pressure

    green_times = allocate_green_times(
        pressures,
        lanes_cfg["intersection"]
    )

    return green_times


if __name__ == "__main__":
    result = run_signal_cycle()
    print("\n🟢 Signal Output:", result)

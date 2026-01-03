def allocate_green_times(pressures, config):
    """
    Allocate green time proportional to pressure
    """
    total_pressure = sum(pressures.values())
    green_times = {}

    min_green = config["min_green"]
    max_green = config["max_green"]
    total_cycle = config["total_cycle_time"]

    if total_pressure == 0:
        # Equal split if no vehicles
        equal_time = total_cycle // len(pressures)
        return {lane: equal_time for lane in pressures}

    for lane, pressure in pressures.items():
        share = pressure / total_pressure
        green_time = int(share * total_cycle)

        green_time = max(min_green, min(green_time, max_green))
        green_times[lane] = green_time

    return green_times

def override_signal(emergency_lane, lanes, override_time):
    """
    Forces emergency lane to GREEN, others RED
    """
    signal_state = {}

    for lane in lanes:
        if lane == emergency_lane:
            signal_state[lane] = override_time
        else:
            signal_state[lane] = 0

    return signal_state


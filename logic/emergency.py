def confirm_emergency(vehicle_counts, confidence, min_conf=0.7):
    """
    Confirms emergency based on:
    - Siren confidence
    - Presence of heavy vehicle (bus/truck)
    """

    heavy_vehicles = vehicle_counts.get("truck", 0) + vehicle_counts.get("bus", 0)

    if heavy_vehicles > 0 and confidence >= min_conf:
        return True

    return False

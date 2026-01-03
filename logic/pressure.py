def calculate_pressure(vehicle_counts, weights):
    """
    Pressure = Σ(count × weight)
    """
    pressure = 0.0

    for vehicle, count in vehicle_counts.items():
        pressure += count * weights.get(vehicle, 0)

    return round(pressure, 2)

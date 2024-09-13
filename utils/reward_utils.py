# Function for exponential moving average (as provided earlier)
def exponential_moving_average(old_value, new_value, alpha=0.8):
    # If old_value is 0, return the new_value as the starting point
    if old_value == 0:
        return new_value
    # Update the exponential moving average
    return alpha * old_value + (1 - alpha) * new_value
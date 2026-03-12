

def round(val: int | float, type: type = None):
    if type is int or isinstance(val, int):
        num_digits = len(str(int(val)))
        round_to = (num_digits - 2) * -1
        if num_digits > 2:
            return round(val, round_to)
        else:
            return int(val)
    elif type is float or isinstance(val, float):
        return round(val, 2)
    else:
        pass
    return val

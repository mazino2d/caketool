from typing import Union


def round(val: Union[int, float], type: type = None):
    if type == int or isinstance(val, int):
        num_digits = len(str(int(val)))
        round_to = (num_digits - 2) * -1
        if num_digits > 2:
            return round(val, round_to)
        else:
            return int(val)
    elif type == float or isinstance(val, int):
        return round(val, 2)
    else:
        pass
    return val

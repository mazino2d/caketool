def round(val: int | float, type: type = None):
    """Round *val* to a human-readable precision based on its magnitude or type.

    For integers, rounds to the nearest power-of-10 appropriate for the
    number of digits (e.g. 1 234 → 1 200, 99 → 99).  For floats, always
    rounds to 2 decimal places.  Falls back to returning *val* unchanged when
    the type cannot be inferred.

    Parameters
    ----------
    val : int | float
        The numeric value to round.
    type : type, optional
        Explicit type hint (``int`` or ``float``).  When ``None``, the type
        is inferred from ``val`` itself.

    Returns
    -------
    int | float
        Rounded value.

    Examples
    --------
    >>> round(12345, int)
    12300
    >>> round(3.14159, float)
    3.14
    >>> round(7, int)
    7
    """
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

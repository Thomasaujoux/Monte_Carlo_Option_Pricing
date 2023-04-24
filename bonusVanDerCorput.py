def van_der_corput(n, base):
    """
    Cette fonction génère une suite de van der Corput de longueur n avec la base donnée.
    """
    result = []
    for i in range(n):
        x = 0
        factor = 1.0 / base
        m = i
        while m > 0:
            x += (m % base) * factor
            m //= base
            factor /= base
        result.append(x)
    return result

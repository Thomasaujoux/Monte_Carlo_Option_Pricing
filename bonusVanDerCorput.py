def van_der_corput_base_2(n):
    """
    Cette fonction génère une suite de van der Corput de longueur n avec la base donnée.
    """
    result = []
    for i in range(n):
        x = 0
        factor = 1.0 / 2
        m = i
        while m > 0:
            x += (m % 2) * factor
            m //= 2
            factor /= 2
        result.append(x)
    return result

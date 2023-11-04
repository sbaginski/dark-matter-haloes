import math


def divide_safely(a, b):
    """
    """
    if b == 0:
        return math.inf
    elif b == math.inf:
        return 0
    else:
        return a / b


def calculate_simpsons_rule(a, b, f):
    return ((b - a) / 6) * (f(a) + 4 * f((a + b) / 2) + f(b))


def integrate(a, b, f, n=10):
    interval_length = (b - a) / n
    result = 0
    left = a
    right = left + interval_length
    while right <= b:
        result += calculate_simpsons_rule(left, right, f)
        left += interval_length
        right += interval_length
    result += calculate_simpsons_rule(left, b, f)
    return result


def get_time(z, h_0=70, omega_r=0, omega_m=0.3, omega_lambda=0.7):
    omega = omega_r + omega_m + omega_lambda

    def f(x):
        return divide_safely(
            1,
            (h_0 * (divide_safely(omega_r, x**2)
                    + divide_safely(omega_m, x)
                    + omega_lambda * x**2
                    + (1 - omega))**(1 / 2))
        )

    a = 1 / (1 + z)
    # Time is expressed in Gyr (giga years)
    time = integrate(0, a, f, 1000000) * 3.086e19 * 1e-9 / (3600 * 24 * 365)
    return time


if __name__ == '__main__':
    z_value = float(input('redshift: '))
    print(f'{get_time(z_value)} Gyr')

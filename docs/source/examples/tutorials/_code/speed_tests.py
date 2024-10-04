import timeit

size = 10_000_000

setup_parabola_python = """
from rand_distr_python import sample_parabola
"""
setup_parabola_numpy = """
from rand_distr_parabola_numpy import sample_parabola
"""

test_parabola = f"""
sample_parabola((-2, 2), {size})
"""

# number = 1
# print(timeit.timeit(setup=setup_parabola_python, stmt=test_parabola, number=number) / number)

# number = 10
# print(timeit.timeit(setup=setup_parabola_numpy, stmt=test_parabola, number=number) / number)


# number = 10
# print(timeit.timeit(
#     setup = """
# import numpy as np;
# from rand_distr_arb_distr import sample_distribution;
# x = np.linspace(-2, 2, 100);
# xedges = np.append(x - (x[1] - x[0]) / 2.0, x[-1] + (x[1] - x[0]) / 2.0);
# y = x**2
#     """,
#     stmt= f"""
# sample_distribution(x, y, {size})
#     """,
#     number = number
# ) / number)

# number = 10
# print(timeit.timeit(
#     setup = """
# import numpy as np;
# from rand_distr_callable import sample_analytic_distribution;
# def f(x):
#     return x**2
#     """,
#     stmt= f"""
# sample_analytic_distribution(f, (-2, 2), {size})
#     """,
#     number = number
# ) / number)

number = 10
print(timeit.timeit(
    setup = """
import numpy as np;
from rand_distr_discrete import sample_discrete_distribution;
x = np.linspace(-2, 2, 100)
y = x**2
    """,
    stmt= f"""
sample_discrete_distribution(x, y, {size})
    """,
    number = number
) / number)


import math


def function(x):
    f = (x - 4) ** 2 + 2 * math.e ** x
    return f


def dfunction(x):
    f = 2 * x + 2 * math.e ** x - 8
    return f


etas = [0.001, 0.0001, 0.00001, 0.000001]
for eta in etas:
    x = 0.0
    i = 0
    while abs(dfunction(x)) > 0.00001:
        x = x - eta * dfunction(x)
    print((eta, x, function(x)))

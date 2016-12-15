from __future__ import print_function
from distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)
print(client)


class Bogus(object):
    def __init__(self, y):
        self.y = y

    def square(self, x):
        return self.y * x ** 2 ** 2 ** 2


def bogus_helper(_args):
    bog, x = _args
    return bog.square(x)


def square(x):
    return x ** 2 ** 2 ** 2


def neg(x):
    return -x

# submit many function calls:
A = client.map(square, range(10))
# print(A)
B = client.map(neg, A)
# print(B)
# submit individual function calls:
total = client.submit(sum, B)
print(total.result())


bg = Bogus(2)
args = [[bg, x] for x in range(10)]
C = client.map(bogus_helper, args)
results = client.gather(C)
print(results)

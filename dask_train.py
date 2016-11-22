from __future__ import print_function
from distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)
print(client)


def square(x):
    return x ** 2 ** 2


def neg(x):
    return -x

# submit many function calls:
A = client.map(square, range(10))
B = client.map(neg, A)
# submit individual function calls:
total = client.submit(sum, B)
print(total.result())

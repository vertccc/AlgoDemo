import ctypes
import numba
so = ctypes.cdll.LoadLibrary

lib = so("./x.so")

# lib.fib(40)  316ms
# fibp(40)     20.5s
# fibp2(40)    160 + 424 ms
# fibb(40)     2.94 us

def fibp(n):
    if n<=2:
        return 1
    return fibp(n-1)+fibp(n-2)


@numba.jit
def fibp2(n):
    if n<=2:
        return 1
    return fibp2(n-1)+fibp2(n-2)


def fibb(n):
    a,b = 1,1
    if n<=2:
        return 1
    for _ in range(n-2):
        a,b = a+b,a
    return a
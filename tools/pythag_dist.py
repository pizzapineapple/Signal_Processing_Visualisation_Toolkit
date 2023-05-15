from math import sqrt
import numpy
def pythag_dist(signal_0, signal_1) :
    if len(signal_0) != len(signal_1) : print("signal length mismatch")
    return sqrt(sum((signal_0-signal_1)**2))

if __name__ == "__main__" :
    a = numpy.array([6,3,5,4,1,2])
    b = numpy.array([3,4,8,9,5,2])
    print(pythag_dist(a,b))
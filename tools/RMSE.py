from math import sqrt
import numpy
def RMSE(data1, data_2) :
    if(len(data1) != len(data_2)) : print("signal length mismatch")
    
    return sqrt( (1/len(data1)) * sum( (abs(data1)-abs(data_2))**2 ) )

if __name__ == "__main__" :
    a = numpy.array([6,3,5,4,1,2])
    b = numpy.array([3,4,8,9,5,2])
    print(RMSE(a,b))
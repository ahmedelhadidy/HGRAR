import matplotlib.pyplot as plt
from model.grar.operator import Operator
from model.grar.operator import OperatorType
import skfuzzy.membership as mf

GT = Operator(OperatorType.GTE, True)
LT = Operator(OperatorType.LTE, True)
EQ = Operator(OperatorType.EQ, True)

def EQ2(x,y):
   xx,c = (x,y) if abs(y) > abs(x) else (y,x)
   return mf.gaussmf(xx,c,0.1 * (abs(x) + abs(y)) /2)


EQ3 = lambda x,y : mf.gaussmf(x-y if abs(y) >= abs(x) else y-x,2,0.2 * (abs(x)+abs(y))/2)

print(EQ2(1, 1.1))



def test_eq():
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    fixed = 10
    E = list(EQ.apply(fixed, x) for x in values)
    plt.figure()
    plt.subplot(311)
    plt.plot(values, E)
    plt.xticks(values)
    plt.show()

def test():
    A=[0,1,2,3,4,5,6,7,8,9,10]

    fixed = 5
    fixed_E=10
    GRF=list(GT.apply(x,fixed) for x in A)
    LRF=list(LT.apply(x,fixed) for x in A)

    print(GRF)
    print(LRF)
    plt.figure()
    #plt.subplot(211)
    #plt.plot(A,GRF)
    plt.subplot(311)
    plt.plot(A,LRF)
    plt.show()


test()
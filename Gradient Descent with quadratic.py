import matplotlib.pyplot as plt

step_amount = 0.01

def f(x):
    return 2*(x**2) + 12*x + 32


def backPropWithOneInput(x, num):
    gradient = 4*x + 12
    for i in range(num):
        print(x,f(x),"gradient:",gradient)
        gradient = 4*x + 12
        x -= step_amount*gradient


backPropWithOneInput(20,1000)

#2x^2 + 12x + 32
#4x + 12 = 0
#-12 = 4x


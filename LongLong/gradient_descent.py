# 2022-2-8 luke 
# demo of gradient descent in linear regression
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n,w,b):
    """generate data for linear regression of length n"""
    data = np.zeros([n,2])
    for i in range(0,n):
        data[i,0] = np.random.rand() * 10 + 5
        data[i,1] = w * data[i,0] + b + np.random.standard_normal()
        # print(np.random.standard_normal())
        # visualize data
    plt.figure("Data")
    plt.scatter(data[:,0],data[:,1],marker='.',color='b',label='data')
    plt.plot([5,15],[5*w+b,15*w+b],'-',color='r',linewidth=2,label='model')
    plt.legend(loc=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return data

def compute_error(b,w,data):
    error = 0
    for x,y in data:
        error += (y - w*x - b) ** 2
    return error/float(len(data))

def gradient_descent(w,b,data,lr):
    """update w and b using gradient descent, with given data, w, b and learning rate lr"""
    w_grad = 0
    b_grad = 0
    N = float(len(data))
    for x,y in data:
        w_grad -= 2/N * (y - w*x - b) * x
        b_grad -= 2/N * (y - w*x - b)
    # update w&b
    w -= lr * w_grad
    b -= lr * b_grad
    return w,b

def GD_runner(data,n=50,lr=0.002):
    """run gradient descent"""
    [w,b] = [0,0]
    history = np.zeros([n,4])
    for i in range(0,n):
        [w,b] = gradient_descent(w,b,data,lr)
        loss = compute_error(b,w,data)
        print(f"[{i+1}/{n}]  loss = {loss}")
        history[i] = [i,loss,w,b]
    return history

def plot_history(history):
    """visualize training process"""
    plt.figure('Loss-Iteration Curve')
    plt.plot(history[:,0],history[:,1])
    # plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    sample = generate_data(200,3,4)
    history = GD_runner(data=sample)
    plot_history(history)
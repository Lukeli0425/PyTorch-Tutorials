from random import sample
import numpy as np

def generate_data(n,w,b):
    """generate data for linear regression of length n"""
    data = np.zeros([n,2])
    for i in range(0,n):
        data[i,0] = np.random.rand() * 10
        data[i,1] = w * data[i,0] + b + np.random.standard_normal()
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

def GD_runner(data,n=200,lr=0.0015):
    [w,b] = [0,0]
    for i in range(0,n):
        [w,b] = gradient_descent(w,b,data,lr)
        print(f"[{i+1}/{n}]  current loss = {compute_error(b,w,data)}")

if __name__ == "__main__":
    sample = generate_data(200,5,1)
    # print(f"Data:\n{data}\n")
    GD_runner(data=sample,n=200)
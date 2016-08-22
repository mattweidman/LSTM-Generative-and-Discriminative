import math
import numpy as np

from LSTM_layer import LSTM_layer, LSTM_layer_gradient

epsilon = 0.00001
num_examples = 1

def get_n(h):
    n = 1
    for dim in h.shape:
        n *= dim
    return n

def mse(h, y):
    return 1/(2*num_examples)*np.sum((h-y)**2, axis=0)

def dmse(h, y):
    return 1/num_examples*(h-y)

def num_elements(x):
    n = 1
    for dim in x.shape:
        n *= dim
    return n

def print_grad(grad):
    dtheta, dx, ds_prev, dh_prev = grad.to_tuple()
    print("gradient difference for theta:")
    for w in dtheta:
        print((w**2).sum()/num_elements(w))
    print("gradient difference for x: " + str((dx**2).sum()/num_elements(dx)))
    print("gradient difference for s_prev: " + str((ds_prev**2).sum()/
        num_elements(ds_prev)))
    print("gradient difference for h_prev: " + str((dh_prev**2).sum()/
        num_elements(dh_prev)))

# loss and h are functions; loss is a function of outp and y
# outp is a function of nothing
# param is what you want to take the numerical gradient with respect to
# outp and y should return the same size
def numerical_gradient_param(loss, outp, y, param):
    grad_param = np.zeros(param.shape)
    for i in range(len(param)):
        for j in range(len(param[i])):
            param[i,j] += epsilon
            loss1 = loss(outp(), y).sum()
            param[i,j] -= 2*epsilon
            loss2 = loss(outp(), y).sum()
            param[i,j] += epsilon
            grad_param[i,j] = (loss1-loss2)/(2*epsilon)
    return grad_param

def numerical_gradient(layer, x, loss, s_prev, h_prev):

    # gradient with respect to theta
    grad_theta = [np.zeros(w.shape) for w in layer.theta]
    for wgt, dwgt in zip(layer.theta, grad_theta):
        for i in range(len(wgt)):
            for j in range(len(wgt[i])):
                wgt[i,j] += epsilon
                loss1 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
                wgt[i,j] -= 2*epsilon
                loss2 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
                wgt[i,j] += epsilon
                dwgt[i,j] = (loss1-loss2)/(2*epsilon)

    # gradient with respect to x
    grad_x = np.zeros(x.shape)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i,j] += epsilon
            loss1 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            x[i,j] -= 2*epsilon
            loss2 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            x[i,j] += epsilon
            grad_x[i,j] = (loss1-loss2)/(2*epsilon)

    # gradient with respect to s_prev
    grad_s_prev = np.zeros(s_prev.shape)
    for i in range(len(s_prev)):
        for j in range(len(s_prev[i])):
            s_prev[i,j] += epsilon
            loss1 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            s_prev[i,j] -= 2*epsilon
            loss2 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            s_prev[i,j] += epsilon
            grad_s_prev[i,j] = (loss1-loss2)/(2*epsilon)

    # gradient with respect to h_prev
    grad_h_prev = np.zeros(h_prev.shape)
    for i in range(len(h_prev)):
        for j in range(len(h_prev[i])):
            h_prev[i,j] += epsilon
            loss1 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            h_prev[i,j] -= 2*epsilon
            loss2 = (layer.forward_prop_once(x, s_prev, h_prev)[1]**2).sum()
            h_prev[i,j] += epsilon
            grad_h_prev[i,j] = (loss1-loss2)/(2*epsilon)

    return LSTM_layer_gradient(grad_theta, grad_x, grad_s_prev, grad_h_prev)

def check_layer():
    input_size = 10
    output_size = 5
    layer = LSTM_layer(input_size, output_size)
    x = np.random.randn(num_examples, input_size)
    y = np.random.randn(num_examples, output_size)
    s_prev = np.random.randn(num_examples, output_size) # test with s_prev
    h_prev = np.random.randn(num_examples, output_size) # and h_prev = 0 too
    grad = layer.backprop(x, lambda h: dmse(h, y), s_prev, h_prev)
    n_grad = numerical_gradient(layer, x, mse, s_prev, h_prev)
    grad_diff = grad.add(n_grad.multiply(-1))
    print_grad(grad_diff)

def check_mse():
    output_size = 10
    h = np.random.randn(num_examples, output_size)
    y = np.random.randn(num_examples, output_size)
    outp_funct = lambda: h
    n_grad = numerical_gradient_param(mse, outp_funct, y, h)
    grad = dmse(h,y)
    grad_diff = grad - n_grad
    print(grad_diff)

if __name__ == "__main__":
    check_mse()

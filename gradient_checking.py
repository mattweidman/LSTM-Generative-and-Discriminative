import math
import numpy as np

from LSTM_layer import LSTM_layer, LSTM_layer_gradient
from LSTM import LSTM

epsilon = 0.00001
num_examples = 1

def get_n(h):
    n = 1
    for dim in h.shape:
        n *= dim
    return n

def mse(h, y):
    return 1/(2*num_examples)*np.sum((h-y)**2)

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

def numerical_gradient(layer, x, y, loss, s_prev, h_prev):

    outp_funct = lambda: layer.forward_prop_once(x, s_prev, h_prev)

    # gradient with respect to theta
    grad_theta = [numerical_gradient_param(mse, outp_funct, y, w)
        for w in layer.theta]

    # gradient with respect to x
    grad_x = numerical_gradient_param(mse, outp_funct, y, x)

    # gradient with respect to s_prev
    grad_s_prev = numerical_gradient_param(mse, outp_funct, y, s_prev)

    # gradient with respect to h_prev
    grad_h_prev = numerical_gradient_param(mse, outp_funct, y, h_prev)

    return LSTM_layer_gradient(grad_theta, grad_x, grad_s_prev, grad_h_prev)

def check_layer():
    input_size = 10
    output_size = 5
    layer = LSTM_layer(input_size, output_size)
    x = np.random.randn(num_examples, input_size)
    y = np.random.randn(num_examples, output_size)
    s_prev = np.random.randn(num_examples, output_size) # test with s_prev
    h_prev = np.random.randn(num_examples, output_size) # and h_prev = 0 too
    dloss = lambda h_: dmse(h_, y)
    grad = layer.backprop(x, dloss, s_prev, h_prev)
    n_grad = numerical_gradient(layer, x, y, mse, s_prev, h_prev)
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

def check_phi():
    veclen = 10
    h = np.random.randn(num_examples, veclen)
    o = np.random.randn(num_examples, veclen)
    y = np.random.randn(num_examples, veclen)
    outp_funct = lambda: h
    loss = lambda h_, y_: np.tanh(h_)*o
    dloss = lambda h_, y_: (1-np.tanh(h_)**2)*o
    n_grad = numerical_gradient_param(loss, outp_funct, y, h)
    grad = dloss(h,y)
    grad_diff = grad-n_grad
    print(grad_diff)

def forward_gates(g, i, f, o, s_prev):
    s = g*i + s_prev.T*f
    h = np.tanh(s)*o
    return h

def backward_gates(g, i, f, o, s_prev, h, y):
    s = g*i + s_prev.T*f
    dLdh = 1/num_examples * (h-y)
    dLdo = dLdh * np.tanh(s)
    dLds = dLdh * o * (1-np.tanh(s)**2)
    dLdg = dLds * i
    dLdi = dLds * g
    dLdf = dLds * s_prev.T
    dLds_prev = dLds * f
    return dLds_prev.T

def check_gates():
    veclen = 10
    nrr = lambda: np.random.randn(num_examples, veclen)
    g = nrr()
    i = nrr()
    f = nrr()
    o = nrr()
    s_prev = nrr().T
    y = nrr()
    outp_funct = lambda: forward_gates(g, i, f, o, s_prev)
    n_grad = numerical_gradient_param(mse, outp_funct, y, s_prev)
    grad = backward_gates(g, i, f, o, s_prev, outp_funct(), y)
    grad_diff = grad - n_grad
    print(grad_diff)

def multiply(W, x):
    return W.dot(x.T)

def back_multiply(W, x):
    return W.sum(axis=0)

def check_multiply():
    in_size = 20
    out_size = 5
    x = np.random.randn(num_examples, in_size)
    W = np.random.randn(out_size, in_size)
    g = multiply(W, x)
    outp_funct = lambda: multiply(W, x)
    loss = lambda h_, y_: h_
    n_grad = numerical_gradient_param(loss, outp_funct, g, x)
    grad = back_multiply(W, x)
    grad_diff = grad - n_grad
    print(grad_diff)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def phi(x):
    return np.tanh(x)

def phi_gate(x, h_prev, Wx, Wh, b):
    return phi(Wx.dot(x.T) + Wh.dot(h_prev.T) + b)

def sigmoid_gate(x, h_prev, Wx, Wh, b):
    return sigmoid(Wx.dot(x.T) + Wh.dot(h_prev.T) + b)

def back_single_gate(x, h_prev, grad_in, W_x, W_h):
    dLdW_x = grad_in.dot(x)
    dLdW_h = grad_in.dot(h_prev)
    dLdb_ = grad_in.sum(axis=1)[:,np.newaxis]
    dLdx_ = W_x.T.dot(grad_in)
    dLdh_ = W_h.T.dot(grad_in)
    return dLdW_x, dLdW_h, dLdb_, dLdx_.T, dLdh_.T

def check_single_gate():
    in_size = 20
    out_size = 5
    x = np.random.randn(num_examples, in_size)
    h_prev = np.random.randn(num_examples, out_size)
    Wx = np.random.randn(out_size, in_size)
    Wh = np.random.randn(out_size, out_size)
    b = np.random.randn(out_size, 1)
    g = phi_gate(x, h_prev, Wx, Wh, b)
    outp_funct = lambda: phi_gate(x, h_prev, Wx, Wh, b)
    loss = lambda h_, y_: h_
    n_grad = numerical_gradient_param(loss, outp_funct, g, h_prev)
    grad_in = 1-g**2
    grad = back_single_gate(x, h_prev, grad_in, Wx, Wh)[4]
    grad_diff = grad - n_grad
    print(grad_diff)

def forward_sep_layer(x, s_prev, h_prev, layer):
    g = phi(Wgx.dot(x.T) + Wgh.dot(h_prev.T) + bg)
    i = sigmoid(Wix.dot(x.T) + Wih.dot(h_prev.T) + bi)
    f = sigmoid(Wfx.dot(x.T) + Wfh.dot(h_prev.T) + bf)
    o = sigmoid(Wox.dot(x.T) + Woh.dot(h_prev.T) + bo)
    s = g*i + s_prev.T*f
    h = phi(s)*o
    return g, i, f, o, s, h

def forward_prop_once(x, s_prev, h_prev, layer, return_gates=False):
    g = phi(layer.Wgx.dot(x.T) + layer.Wgh.dot(h_prev.T) + layer.bg)
    i = sigmoid(layer.Wix.dot(x.T) + layer.Wih.dot(h_prev.T) + layer.bi)
    f = sigmoid(layer.Wfx.dot(x.T) + layer.Wfh.dot(h_prev.T) + layer.bf)
    o = sigmoid(layer.Wox.dot(x.T) + layer.Woh.dot(h_prev.T) + layer.bo)
    s = g*i + s_prev.T*f
    h = phi(s)*o
    if return_gates:
        return s.T, h.T, (g, i, f, o, s, h)
    else:
        return s.T, h.T

def backward_sep_layer(x, s_prev, h_prev, layer, dloss, s_next_grad=None,
        h_next_grad=None, gate_values=None):

    # default values for s_next_grad and h_next_grad
    if s_next_grad is None:
        s_next_grad = np.zeros(s_prev.shape)
    if h_next_grad is None:
        h_next_grad = np.zeros(h_prev.shape)

    # propagate forward
    if gate_values is None:
        _, _, (g, i, f, o, s, h) = forward_prop_once(x, s_prev, h_prev, layer,
            return_gates=True)
    else:
        g, i, f, o, s, h = gate_values
    assert g.shape[0] == h.shape[0]
    assert i.shape[1] == h.shape[1]
    assert f.shape[0] == s.shape[0]
    assert o.shape[1] == s.shape[1]

    # backprop to each gates
    dLdh = dloss(h.T).T + h_next_grad.T
    dLdo = dLdh * phi(s)
    dLds = dLdh * o * (1-phi(s)**2) + s_next_grad.T
    dLdg = dLds * i
    dLdi = dLds * g
    dLdf = dLds * s_prev.T
    dLds_prev = dLds * f

    # finds dL/dW?x, dL/dW?h, dL/db?, dL/dx_?, and dL/dh_?
    # grad_in is equal to dL/d? * sigmoid_prime or dL/d? * phi_prime
    # ? indicates the name of a gate - g, i, f, or o
    def backprop_gate(grad_in, W_x, W_h):
        dLdW_x = grad_in.dot(x)
        dLdW_h = grad_in.dot(h_prev)
        dLdb_ = grad_in.sum(axis=1)[:,np.newaxis]
        dLdx_ = W_x.T.dot(grad_in)
        dLdh_ = W_h.T.dot(grad_in)
        return dLdW_x, dLdW_h, dLdb_, dLdx_, dLdh_

    # backprop within each gate
    g_prime = dLdg * (1-g**2)
    dLdWgx, dLdWgh, dLdbg, dLdxg, dLdhg = backprop_gate(g_prime, layer.Wgx,
        layer.Wgh)
    i_prime = dLdi * i*(1-i)
    dLdWix, dLdWih, dLdbi, dLdxi, dLdhi = backprop_gate(i_prime, layer.Wix,
        layer.Wih)
    f_prime = dLdf * f*(1-f)
    dLdWfx, dLdWfh, dLdbf, dLdxf, dLdhf = backprop_gate(f_prime, layer.Wfx,
        layer.Wfh)
    o_prime = dLdo * o*(1-o)
    dLdWox, dLdWoh, dLdbo, dLdxo, dLdho = backprop_gate(o_prime, layer.Wox,
        layer.Woh)

    # combine everything into one data structure
    dLdtheta = [dLdWgx, dLdWix, dLdWfx, dLdWox, dLdWgh, dLdWih, dLdWfh,
       dLdWoh, dLdbg, dLdbi, dLdbf, dLdbo]
    dLdx = dLdxg + dLdxi + dLdxf + dLdxo
    dLdh_prev = dLdhg + dLdhi + dLdhf + dLdho

    return LSTM_layer_gradient(dLdtheta, dLdx.T, dLds_prev.T, dLdh_prev.T)

def check_sep_layer():
    input_size = 10
    output_size = 5
    x = np.random.randn(num_examples, input_size)
    s_prev = np.random.randn(num_examples, output_size)
    h_prev = np.random.randn(num_examples, output_size)
    s_next_grad = np.random.randn(num_examples, output_size)
    h_next_grad = np.random.randn(num_examples, output_size)
    y = np.random.randn(num_examples, output_size)
    layer = LSTM_layer(input_size, output_size)

    outp_funct = lambda: layer.forward_prop_once(x, s_prev, h_prev)[1]
    n_grad_theta = []
    for w in layer.theta:
        n_grad_theta.append(numerical_gradient_param(mse, outp_funct, y, w))
    n_grad_x = numerical_gradient_param(mse, outp_funct, y, x)
    n_grad_s_prev = numerical_gradient_param(mse, outp_funct, y, s_prev)
    n_grad_h_prev = numerical_gradient_param(mse, outp_funct, y, h_prev)

    dloss = lambda h_: dmse(h_, y)
    grad = layer.backprop(x, dloss, s_prev, h_prev).to_tuple()

    print("theta gradient:")
    for wn, wg in zip(n_grad_theta, grad[0]):
        print(((wn-wg)**2).sum())
    print("x gradient: ", ((n_grad_x-grad[1])**2).sum())
    print("s_prev gradient: ", ((n_grad_s_prev-grad[2])**2).sum())
    print("h_prev gradient: ", ((n_grad_h_prev-grad[3])**2).sum())

def check_multiple_layers():
    input_size = 10
    hidden_size = 7
    output_size = 5
    x = np.random.randn(num_examples, input_size)
    s_prev = [np.random.randn(num_examples, hidden_size),
        np.random.randn(num_examples, output_size)]
    h_prev = [np.random.randn(num_examples, hidden_size),
        np.random.randn(num_examples, output_size)]
    y = np.random.randn(num_examples, output_size)
    lstm = LSTM()
    lstm.add_layer(LSTM_layer(input_size, hidden_size))
    lstm.add_layer(LSTM_layer(hidden_size, output_size))

    outp_funct = lambda: lstm.forward_prop_once(x, s_prev, h_prev)[1][-1]
    n_grad_theta = []
    for layer in lstm.layers:
        n_grad_theta_l = [numerical_gradient_param(mse, outp_funct, y, w)
            for w in layer.theta]
        n_grad_theta.append(n_grad_theta_l)
    n_grad_x = numerical_gradient_param(mse, outp_funct, y, x)
    n_grad_s_prev0 = numerical_gradient_param(mse, outp_funct, y, s_prev[0])
    n_grad_s_prev1 = numerical_gradient_param(mse, outp_funct, y, s_prev[1])
    n_grad_h_prev0 = numerical_gradient_param(mse, outp_funct, y, h_prev[0])
    n_grad_h_prev1 = numerical_gradient_param(mse, outp_funct, y, h_prev[1])

    grads = lstm.backprop_once(x, y, dmse, s_prev, h_prev)

    print("theta gradient:")
    for ngt_l, g in zip(n_grad_theta, grads):
        for nw, w in zip(ngt_l, g.dLdtheta):
            print(((nw-w)**2).sum())
    print("x gradient: ", ((n_grad_x-grads[0].dLdx)**2).sum())
    print("s_prev[0] gradient: ", ((n_grad_s_prev0-grads[0].dLds_prev)**2).sum())
    print("s_prev[1] gradient: ", ((n_grad_s_prev1-grads[1].dLds_prev)**2).sum())
    print("h_prev[0] gradient: ", ((n_grad_h_prev0-grads[0].dLdh_prev)**2).sum())
    print("h_prev[1] gradient: ", ((n_grad_h_prev1-grads[1].dLdh_prev)**2).sum())

def check_BPTT():
    input_size = 10
    hidden_size = 7
    output_size = 5
    seq_len = 20
    x = np.random.randn(num_examples, seq_len, input_size)
    y = np.random.randn(num_examples, seq_len, output_size)
    nz = lambda width: np.zeros((num_examples, width))
    s_prev = [nz(hidden_size), nz(output_size)]
    h_prev = [nz(hidden_size), nz(output_size)]
    lstm = LSTM()
    lstm.add_layer(LSTM_layer(input_size, hidden_size))
    lstm.add_layer(LSTM_layer(hidden_size, output_size))

    outp_funct = lambda: lstm.forward_prop(x)
    n_grad_theta = []
    for layer in lstm.layers:
        n_grad_theta_l = [numerical_gradient_param(mse, outp_funct, y, w)
            for w in layer.theta]
        n_grad_theta.append(n_grad_theta_l)
    # n_grad_x = numerical_gradient_param(mse, outp_funct, y, x[0])

    grads = lstm.BPTT(x, y, dmse)

    print("theta gradient:")
    for ngt_l, g in zip(n_grad_theta, grads):
        for nw, w in zip(ngt_l, g.dLdtheta):
            print(((nw-w)**2).sum())
    # print("x gradient: ", ((n_grad_x-grads[0].dLdx)**2).sum())
    # print("n_grad_x: ", n_grad_x)
    # print("grads[0].dLdx: ", grads[0].dLdx)
    # print("n_grad_x.shape: ", n_grad_x.shape)
    # print("grads[0].dLdx.shape: ", grads[0].dLdx.shape)
    # print("(n_grad_x-grads[0].dLdx).shape: ", (n_grad_x-grads[0].dLdx).shape)

if __name__ == "__main__":
    check_BPTT()

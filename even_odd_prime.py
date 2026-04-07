"""
For each number 1-100

if the number is prime, print 'prime'
if the number is divisible by 2, print 'even'
if the number is not divisible by 2, print 'odd'
"""

from typing import List

import numpy as np
from nn_library.train import train
from nn_library.nn import NeuralNet
from nn_library.layers import Linear, Tanh
from nn_library.optimizer import SGD

def is_prime(n:int):
    if n < 2:
        return False
    
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    
    return True

def even_odd_prime_encode(x:int) -> List[int]:
    if is_prime(x)==True:
        return [0,0,1]
    elif x % 2 == 0:
        return [0,1,0]
    elif x%2 != 0:
        return [1,0,0]

def binary_encode(x:int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    even_odd_prime_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=3)
])

train(net, inputs, targets, num_epochs=5000, optimizer=SGD(lr=0.001))

for x in range(1, 101):
    prediction = net.forward(binary_encode(x))
    prediction_idx = np.argmax(prediction)
    actual_idx = np.argmax(even_odd_prime_encode(x))
    labels = ['odd','even','prime']
    print(x, labels[prediction_idx], labels[actual_idx])
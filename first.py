import math
import random
from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.nn import Layer

layer = Layer(3, 4) # 3 input 4 nöron

# layerları random ağırlıkla dolduralım
for neuron in layer.neurons:
    neuron.b = Value(1.0)
    for w in neuron.w:
        w.data = random.uniform(0.001, 1.0)
    

# girdi
x = [Value(0.4), Value(0.3), Value(1.4)]

# beklediğimiz çıktı
target = [1.0, 1.0, 1.0, 1.0]

train_cycles = 10

learning_rate = 0.01

while train_cycles > 0:

    print(train_cycles, " train cycle(s) left")


    # w*x - b
    y = layer(x)

    # sapmalar
    losses = []

    for i, yi in enumerate(y):
        print("neuron: ", i, " result: ", yi.data)
        loss = (yi - target[i]) ** 2
        losses.append(loss)

    loss = sum(losses)
    loss.backward()

    print("        ")

    # print results
    for i, neuron in enumerate(layer.neurons):
        print("neuron: ", i)

        for j, w in enumerate(neuron.w):
            print("connection: ", j, " weight: ", w.data, " gradient: ", w.grad)
            # gradient descent olarak çıkan graientin tersine doğru eğitiyoruz
            w.data -= w.grad * learning_rate


    print("        ")
    
    train_cycles -= 1

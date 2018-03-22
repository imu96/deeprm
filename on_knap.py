import numpy as np
import parameters
from math import exp

max_res = 200
max_density = 20
min_density = 1

def psi(upper_bound, lower_bound, frac):
    threshold = ((upper_bound*exp(1)/lower_bound)**frac)*(lower_bound/exp(1))
    return threshold

#item structure: array  [int weight, int density]

def on_kp_threshold(items,  capacity, upper_bound, lower_bound):
    bag = []
    used_space = 0
    for i in items:
        if  used_space == capacity:
            break;
        if i[0] + used_space > capacity:
            continue;
        frac = used_space*1.0 / capacity
        psi_i = psi(upper_bound, lower_bound, frac);
        if i[1] >=  psi_i:
            bag.append(i);
            used_space += i[0]
    return bag;

def value(bag):
    val = 0
    for i in bag:
        val += i[0] * i[1]
    return val

test_upper = 200
test_lower = 1
test_capacity = 21
test_cap2 = 40

test_items = []
test_items.append([20, 20])
test_items.append([1, 100])
test_items.append([30, 1])

#print on_kp_threshold(test_items,test_cap2, test_upper, test_lower)




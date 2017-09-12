from FP import List, Nothing
import numpy as np

def list_to_tuple(v, acc):
    return acc + (v,)

def split_to_tuple(v):
    return (np.array(v[0]), np.array(v[1]))

def nothing_if_empty(value):
    return Nothing() if not value else List(value)

def is_not_none(value):
    return value is not None

def add_reducer(v, acc):
    return acc + v
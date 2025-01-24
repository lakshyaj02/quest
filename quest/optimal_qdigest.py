import sys
import numpy as np
import math
from qdigest import QuantileDigest

class UnaryNode:
    def __init__(self, value):
        self.value = value
        self.child = None

    def get_child(self):
        if self.child is None:
            self.child = UnaryNode(self.value)
        return self.child

class OptimalQDigest:
    def __init__(self, epsilon) -> None:
        self.qdigest_list = [QuantileDigest(epsilon)]
        self.epsilon = epsilon
        self.k = int(math.log(self.epsilon*self.qdigest_list[0].MAX_BITS)) - 1
        self.qdigest_list = [QuantileDigest(epsilon) for _ in range(self.k - 1)]

    def insert(self, value):
        pass
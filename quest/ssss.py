from typing import Generic, TypeVar, Dict, List, Tuple, Hashable
from dataclasses import dataclass
import hashlib
import random
import numpy as np

L = TypeVar('L', bound=Hashable)
I = TypeVar('I', bound=Hashable)

@dataclass
class Config:
    max_num_counters: int
    cardinality_sketch_config: 'HyperLogLogConfig'
    seeds: List[int]

    @classmethod
    def new(cls, max_num_counters: int, cardinality_sketch_config: 'HyperLogLogConfig', seeds: List[int] = None):
        if max_num_counters == 0:
            raise ValueError("ZeroMaxNumCounters")
        if seeds is None:
            seeds = [random.randint(0, 2**64 - 1) for _ in range(4)]
        return cls(max_num_counters, cardinality_sketch_config, seeds)

@dataclass
class HyperLogLogConfig:
    num_registers: int
    seeds: List[int]

class HyperLogLog:
    def __init__(self, config: HyperLogLogConfig):
        self.config = config
        self.registers = [0] * config.num_registers
        self.num_zero_registers = config.num_registers
        self.z_inv = float(config.num_registers)

    def insert(self, item: I):
        h = self.hash_item(item)
        idx = h & (self.config.num_registers - 1)
        val = (h >> self.config.num_registers).bit_length()
        if self.registers[idx] < val:
            if self.registers[idx] == 0:
                self.num_zero_registers -= 1
            self.z_inv -= 2.0 ** -self.registers[idx]
            self.z_inv += 2.0 ** -val
            self.registers[idx] = val

    def cardinality(self) -> int:
        alpha = 0.7213 / (1 + 1.079 / self.config.num_registers)
        estimate = (self.config.num_registers ** 2 * alpha) / self.z_inv
        if estimate <= 2.5 * self.config.num_registers:
            if self.num_zero_registers > 0:
                estimate = self.config.num_registers * np.log(self.config.num_registers / self.num_zero_registers)
        return int(estimate)

    def merge(self, other: 'HyperLogLog'):
        if self.config != other.config:
            raise ValueError("ConfigMismatch")
        self.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]
        self.num_zero_registers = sum(1 for r in self.registers if r == 0)
        self.z_inv = sum(2.0 ** -r for r in self.registers)

    def hash_item(self, item: I) -> int:
        h = hashlib.sha256(str(item).encode()).hexdigest()
        return int(h, 16)

class SamplingSpaceSavingSets(Generic[L, I]):
    def __init__(self, config: Config):
        self.config = config
        self.counters: Dict[L, HyperLogLog] = {}
        self.threshold = 0

    def insert(self, label: L, item: I):
        if label in self.counters:
            self.counters[label].insert(item)
        elif len(self.counters) < self.config.max_num_counters:
            self.counters[label] = HyperLogLog(self.config.cardinality_sketch_config)
            self.counters[label].insert(item)
        else:
            cardinality_estimate = self.cardinality_estimate(label, item)
            if cardinality_estimate > self.threshold:
                min_label, min_cardinality = min(
                    ((l, c.cardinality()) for l, c in self.counters.items()),
                    key=lambda x: x[1]
                )
                if cardinality_estimate > min_cardinality:
                    del self.counters[min_label]
                    self.counters[label] = HyperLogLog(self.config.cardinality_sketch_config)
                    self.counters[label].insert(item)
                    self.threshold = min_cardinality

    def merge(self, other: 'SamplingSpaceSavingSets[L, I]'):
        if self.config != other.config:
            raise ValueError("ConfigMismatch")
        
        for label, counter in other.counters.items():
            if label in self.counters:
                self.counters[label].merge(counter)
            else:
                self.counters[label] = counter

        if len(self.counters) > self.config.max_num_counters:
            sorted_counters = sorted(
                ((l, c.cardinality()) for l, c in self.counters.items()),
                key=lambda x: x[1],
                reverse=True
            )
            self.counters = {l: self.counters[l] for l, _ in sorted_counters[:self.config.max_num_counters]}
        
        self.threshold = min(c.cardinality() for c in self.counters.values()) if self.counters else 0

    def cardinality(self, label: L) -> int:
        return self.counters[label].cardinality() if label in self.counters else 0

    def top(self, k: int) -> List[Tuple[L, int]]:
        return sorted(
            ((l, c.cardinality()) for l, c in self.counters.items()),
            key=lambda x: x[1],
            reverse=True
        )[:k]

    def cardinality_estimate(self, label: L, item: I) -> int:
        h = hashlib.sha256(str(item).encode()).hexdigest()
        return int((2**64 - 1) / int(h, 16))

    def num_counters(self) -> int:
        return len(self.counters)
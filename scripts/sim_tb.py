import itertools as it
import json
import operator
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray

from scripts import compress

def g4_mul_int(x, y):
    a = (x & 0x2) >> 1
    b = (x & 0x1)
    c = (y & 0x2) >> 1
    d = (y & 0x1)
    e = (a ^ b) & (c ^ d)
    p = (a & c) ^ e
    q = (b & d) ^ e
    return ( (p<<1) | q )

def g4_mul_bit(x0, x1, y0, y1):
    a = x1
    b = x0
    c = y1
    d = y0
    e = (a ^ b) & (c ^ d)
    p = (a & c) ^ e
    q = (b & d) ^ e
    return q, p

def G4_scl_N_int(x):
    a = (x & 0x2) >> 1 
    b = (x & 0x1)
    p = b
    q = a ^ b
    return ( (p<<1) | q )

def g16_mul_int(x, y):
    a = (x & 0xC) >> 2 
    b = (x & 0x3)
    c = (y & 0xC) >> 2 
    d = (y & 0x3)
    e = G4_mul_int( a ^ b, c ^ d )
    e = G4_scl_N_int(e)
    p = G4_mul( a, c ) ^ e
    q = G4_mul( b, d ) ^ e
    return ( (p<<2) | q )

def g16_mul_bit(x0, x1, x2, x3, y0, y1, y2, y3):
    g16 = g16_mul_int(
            x0 | (x1 << 1) | (x2 << 2) | (x3 << 3),
            y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)
            )
    z0 = g16 & 0x1
    z1 = (g16 >> 1) & 0x1
    z2 = (g16 >> 2) & 0x1
    z3 = (g16 >> 3) & 0x1
    return [z0,z1,z2,z3]

class NewCircuit:
    OP_MAP = {
            compress.OP_XOR: lambda x, y: [x ^ y],
        compress.OP_XNOR: lambda x, y: [not (x ^ y)],
        compress.OP_AND: lambda x, y: [x & y],
        compress.OP_NOT: lambda x: [not x],
        compress.Operation.from_symbol("G4_mul"): g4_mul_bit,
        compress.Operation.from_symbol("G16_mul"): g16_mul_bit
    }

    def __init__(self, fname: str):
        with open(fname, "r") as f:
            s = f.read()
        self.circuit = compress.Circuit.from_circuit_str(s)

    def evaluate(
        self, inputs: dict[compress.Variable, bool]
    ) -> dict[compress.Variable, bool]:
        res = {var: inputs[var] for var in self.circuit.inputs}
        for computations in self.circuit.computations:
            computation = next(iter(computations))
            x = self.OP_MAP[computation.operation](*(res[op] for op in computation.operands))
            for var, output in zip(computation.outputs, x):
                res[var] = output
        return res


class DutWrapper:
    TEST_ITER = 1000

    def __init__(self, circuit, dut):
        self.dut = dut
        self.rnd_handles = {
            int(name[3:]): getattr(dut, name)
            for name in sorted(dir(dut))
            if name.startswith("rnd")
        }
        self.input_handles = {
            name: getattr(dut, name)
            for name in sorted(circuit.circuit.inputs, key=lambda x: x[1:])
        }
        self.output_handles = {
            name: getattr(dut, name)
            for name in sorted(circuit.circuit.outputs, key=lambda x: x[1:])
        }
        self.set_handles = self.input_handles | self.rnd_handles

    def exhaustive_test_niter(self):
        return 2 ** sum(hnd.value.n_bits for hnd in self.set_handles.values())

    def random_pattern(self):
        return (
            {
                name: random.getrandbits(hnd.value.n_bits)
                for name, hnd in self.input_handles.items()
            },
            {
                name: random.getrandbits(hnd.value.n_bits)
                for name, hnd in self.rnd_handles.items()
            },
        )

    def exhaustive_patterns(self):
        def named_patterns(handles):
            for pattern in it.product(
                *(range(2**hnd.value.n_bits) for hnd in handles.values())
            ):
                yield dict(zip(handles.keys(), pattern))

        yield from it.product(
            named_patterns(self.input_handles), named_patterns(self.rnd_handles)
        )

    def test_patterns(self):
        if self.exhaustive_test_niter() <= self.TEST_ITER:
            yield from self.exhaustive_patterns()
        else:
            for _ in range(self.TEST_ITER):
                yield self.random_pattern()

    def reset_inputs(self):
        for hnd in self.input_handles.values():
            hnd.value = LogicArray("X" * hnd.value.n_bits)

    def reset_rnd(self):
        for hnd in self.rnd_handles.values():
            hnd.value = LogicArray("X" * hnd.value.n_bits)

    def reset(self):
        self.reset_inputs()
        self.reset_rnd()

    def apply_pattern(self, pattern):
        for name, x in pattern.items():
            self.set_handles[name].value = x

    def pattern_input_unmasked(self, pattern):
        return {name: self.unmask(x) for name, x in pattern.items()}

    @staticmethod
    def unmask(x):
        return bin(x).count("1") % 2

    def outputs_unmasked(self):
        return {
            name: self.unmask(hnd.value) for name, hnd in self.output_handles.items()
        }


@cocotb.test()
async def test_dut(dut):
    # precompute all possible input/output states
    stats_file = os.environ["STATS"]
    with open(stats_file) as f:
        stats = json.load(f)
    latency = stats["Latency"]
    circuit = NewCircuit(os.environ["CIRCUIT_FILE_PATH"])
    dut_wrapper = DutWrapper(circuit, dut)

    clock = Clock(dut.clk, 10)  # Create a 10us period clock on port clk
    cocotb.start_soon(clock.start())  # Start the clock

    dut_wrapper.reset()
    for _ in range(2):
        await RisingEdge(dut.clk)

    for in_pattern, rnd_pattern in dut_wrapper.test_patterns():
        dut_wrapper.apply_pattern(in_pattern)

        for clkcnt in range(latency + 1):
            if clkcnt == 1:
                dut_wrapper.reset_inputs()
            dut_wrapper.reset_rnd()
            if x := rnd_pattern.get(clkcnt) is not None:
                dut_wrapper.apply_pattern({clkcnt: x})

            await RisingEdge(dut.clk)

        i_umsk = dut_wrapper.pattern_input_unmasked(in_pattern)
        o_umsk = dut_wrapper.outputs_unmasked()
        eval_circuit = circuit.evaluate(i_umsk)
        for name, val in o_umsk.items():
            assert val == eval_circuit[name]

        dut_wrapper.reset()

        for _ in range(2):
            await RisingEdge(dut.clk)

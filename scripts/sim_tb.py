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


class NewCircuit:
    OP_MAP = {
        compress.OP_XOR: (operator.__xor__,),
        compress.OP_XNOR: (lambda x, y: not (x ^ y),),
        compress.OP_AND: (operator.__and__,),
        compress.OP_NOT: (operator.__not__,),
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
            for var, fn in zip(computation.outputs, self.OP_MAP[computation.operation]):
                res[var] = fn(*(res[op] for op in computation.operands))
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

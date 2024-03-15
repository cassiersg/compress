import itertools as it
import json
import os
import random
import logging

logging.getLogger().setLevel(logging.INFO)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray

from scripts import compress
from scripts import circuit_eval


class DutWrapper:
    TEST_ITER = 1000

    def __init__(self, circuit, dut):
        self.dut = dut
        self.rnd_handles = {
            int(name[4:].replace("m", "-")): getattr(dut, name)
            for name in sorted(dir(dut))
            if name.startswith("rnd")
        }

        def handles(vars):
            return {
                name: getattr(dut, name) for name in sorted(vars, key=lambda x: x[1:])
            }

        self.input_handles = handles(circuit.circuit.inputs)
        self.control_handles = handles(circuit.circuit.controls)
        self.output_handles = handles(circuit.circuit.outputs)
        self.set_handles = self.input_handles | self.rnd_handles | self.control_handles

    def exhaustive_test_niter(self):
        return 2 ** sum(hnd.value.n_bits for hnd in self.set_handles.values())

    @staticmethod
    def handles_pattern(handles):
        return {
            name: random.getrandbits(hnd.value.n_bits) for name, hnd in handles.items()
        }

    def random_pattern(self):
        return (
            self.handles_pattern(self.input_handles),
            self.handles_pattern(self.rnd_handles),
            self.handles_pattern(self.control_handles),
        )

    def exhaustive_patterns(self):
        def named_patterns(handles):
            for pattern in it.product(
                *(range(2**hnd.value.n_bits) for hnd in handles.values())
            ):
                yield dict(zip(handles.keys(), pattern))

        yield from it.product(
            named_patterns(self.input_handles),
            named_patterns(self.rnd_handles),
            named_patterns(self.control_handles),
        )

    def test_patterns(self):
        if self.exhaustive_test_niter() <= self.TEST_ITER:
            yield from self.exhaustive_patterns()
        else:
            for _ in range(self.TEST_ITER):
                yield self.random_pattern()

    def reset_inputs(self):
        for hnd in (self.input_handles | self.control_handles).values():
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
    def unmask(x) -> bool:
        return bool(bin(x).count("1") % 2)

    def outputs_unmasked(self) -> dict[compress.Variable, bool]:
        return {
            compress.Variable(name): self.unmask(hnd.value)
            for name, hnd in self.output_handles.items()
        }


# fmt: off
AES_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]
# fmt: on


def aes_sbox_ref(i_umsk):
    x = sum(int(i_umsk[f"i{i}"]) << i for i in range(8))
    x = AES_SBOX[x]
    return {f"o{i}": (x >> i) & 0x1 for i in range(8)}


REF_CIRCUITS = {
    "aes_sbox": aes_sbox_ref,
}


@cocotb.test()
async def test_dut(dut):
    # precompute all possible input/output states
    stats_file = os.environ["STATS"]
    check_eval = os.environ.get("TB_CHECK_EVAL", "1") != "0"
    check_ref = os.environ.get("TB_CHECK_REF")
    if check_eval:
        logging.info("Checking output against the .txt COMPRESS circuit.")
    if check_ref is not None:
        logging.info(f"Checking output to be the {check_ref} function.")
    if not check_eval and check_ref is None:
        logging.warning("Circuit output not verified.")
    with open(stats_file) as f:
        stats = json.load(f)
    latency = stats["Latency"]
    with open(os.environ["CIRCUIT_FILE_PATH"], "r") as f:
        circuit = circuit_eval.CircuitEval(f.read())
    dut_wrapper = DutWrapper(circuit, dut)

    clock = Clock(dut.clk, 10)  # Create a 10us period clock on port clk
    cocotb.start_soon(clock.start())  # Start the clock

    dut_wrapper.reset()
    for _ in range(2):
        await RisingEdge(dut.clk)

    start_cycle = min([0, *dut_wrapper.rnd_handles.keys()])
    for in_pattern, rnd_pattern, ctrl_pattern in dut_wrapper.test_patterns():
        for clkcnt in range(start_cycle, latency + 1):
            if clkcnt == 0:
                dut_wrapper.apply_pattern(in_pattern | ctrl_pattern)
            elif clkcnt == 1:
                dut_wrapper.reset_inputs()
            dut_wrapper.reset_rnd()
            if x := rnd_pattern.get(clkcnt) is not None:
                dut_wrapper.apply_pattern({clkcnt: x})

            await RisingEdge(dut.clk)

        i_umsk = dut_wrapper.pattern_input_unmasked(in_pattern)
        o_umsk = dut_wrapper.outputs_unmasked()
        if check_eval:
            ctrl_pattern_bool = {k: [False, True][v] for k, v in ctrl_pattern.items()}
            eval_circuit = circuit.evaluate(i_umsk, ctrl_pattern_bool)
            for name, val in o_umsk.items():
                assert val == eval_circuit[name]
        if check_ref is not None:
            ref_circuit = REF_CIRCUITS[check_ref](i_umsk)
            for name, val in o_umsk.items():
                assert val == ref_circuit[name]

        dut_wrapper.reset()

        for _ in range(2):
            await RisingEdge(dut.clk)

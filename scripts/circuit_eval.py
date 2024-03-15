import logging
from . import compress


def no_param(f):
    return lambda _, x: f(*x)


class CircuitEval:
    OP_MAP = {
        compress.OP_XOR: no_param(lambda x, y: [x ^ y]),
        compress.OP_XNOR: no_param(lambda x, y: [not (x ^ y)]),
        compress.OP_AND: no_param(lambda x, y: [x & y]),
        compress.OP_NOT: no_param(lambda x: [not x]),
    }

    def __init__(self, s: str):
        self.circuit = compress.Circuit.from_circuit_str(s)
        self.circuit.sort_computations()

    @classmethod
    def add_op(cls, name, fn):
        cls.OP_MAP[name.lower()] = fn

    def evaluate(
        self,
        inputs: dict[compress.Variable, bool],
        controls: dict[compress.Variable, int],
    ) -> dict[compress.Variable, bool]:
        res = {var: inputs[var] for var in self.circuit.inputs}
        for computations in self.circuit.computations:
            computation = next(iter(computations))
            logging.info(f"evaluate {computation}")
            x = self.OP_MAP[computation.operation](
                tuple(controls[c] for c in computation.controls),
                tuple(res[op] for op in computation.operands),
            )
            for var, output in zip(computation.outputs, x):
                res[var] = output
        return res


def g4_mul_int(x, y):
    a = (x & 0x2) >> 1
    b = x & 0x1
    c = (y & 0x2) >> 1
    d = y & 0x1
    e = (a ^ b) & (c ^ d)
    p = (a & c) ^ e
    q = (b & d) ^ e
    return (p << 1) | q


def g4_mul_bit(x0, x1, y0, y1):
    a = x1
    b = x0
    c = y1
    d = y0
    e = (a ^ b) & (c ^ d)
    p = (a & c) ^ e
    q = (b & d) ^ e
    return q, p


def g4_scl_N_int(x):
    a = (x & 0x2) >> 1
    b = x & 0x1
    p = b
    q = a ^ b
    return (p << 1) | q


def g16_mul_int(x, y):
    a = (x & 0xC) >> 2
    b = x & 0x3
    c = (y & 0xC) >> 2
    d = y & 0x3
    e = g4_mul_int(a ^ b, c ^ d)
    e = g4_scl_N_int(e)
    p = g4_mul_int(a, c) ^ e
    q = g4_mul_int(b, d) ^ e
    return (p << 2) | q


def g16_mul_bit(x0, x1, x2, x3, y0, y1, y2, y3):
    g16 = g16_mul_int(
        x0 | (x1 << 1) | (x2 << 2) | (x3 << 3), y0 | (y1 << 1) | (y2 << 2) | (y3 << 3)
    )
    z0 = g16 & 0x1
    z1 = (g16 >> 1) & 0x1
    z2 = (g16 >> 2) & 0x1
    z3 = (g16 >> 3) & 0x1
    return [z0, z1, z2, z3]


def mux2(param, inputs):
    (s,) = param
    in_false, in_true = inputs
    if s == 0:
        return [in_false]
    else:
        return [in_true]


CircuitEval.add_op("G4_mul", no_param(g4_mul_bit))
CircuitEval.add_op("G16_mul", no_param(g16_mul_bit))
CircuitEval.add_op("MUX2", mux2)

from . import compress


class CircuitEval:
    OP_MAP = {
        compress.OP_XOR: lambda x, y: [x ^ y],
        compress.OP_XNOR: lambda x, y: [not (x ^ y)],
        compress.OP_AND: lambda x, y: [x & y],
        compress.OP_NOT: lambda x: [not x],
    }

    def __init__(self, s: str):
        self.circuit = compress.Circuit.from_circuit_str(s)

    def evaluate(
        self, inputs: dict[compress.Variable, bool]
    ) -> dict[compress.Variable, bool]:
        res = {var: inputs[var] for var in self.circuit.inputs}
        for computations in self.circuit.computations:
            computation = next(iter(computations))
            x = self.OP_MAP[computation.operation](
                *(res[op] for op in computation.operands)
            )
            for var, output in zip(computation.outputs, x):
                res[var] = output
        return res

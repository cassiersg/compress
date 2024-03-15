# SPDX-FileCopyrightText: SIMPLE-Crypto contributors
# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2023 SIMPLE-Crypto contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import itertools as it
from pathlib import Path
import random

import numpy as np


class Circuit:
    def __init__(self, inputs, outputs, flex=False, flex_bits=3):
        self.var_map = dict()
        self.inputs = inputs
        self.outputs = outputs
        self.ios = set(inputs) | set(outputs)
        self.flex = flex
        self.flex_bits = flex_bits
        for x in self.ios:
            self.var(x)
        self.ops = []

    def var(self, x):
        while self.var_map.get(x, x) != x:
            x = self.var_map[x]
        return x

    def assign(self, x, y):
        if x in self.ios and y in self.ios:
            raise ValueError(f"Both {x} and {y} are circuit I/Os.")
        elif y in self.ios:
            x, y = y, x
        # replace y by x if they do not already map to each other
        if self.var(x) != self.var(y):
            self.var_map[y] = self.var(x)

    def apply_var_map(self):
        self.ops = [
            (self.var(dest), self.var(op1), self.var(op2), op)
            for dest, op1, op2, op in self.ops
        ]

    def write_op(self, dest, op1, op2, op):
        self.ops.append((dest, op1, op2, op))

    def and_op(self, dest, op1, op2):
        self.write_op(dest, op1, op2, "&")

    def xor_op(self, dest, op1, op2):
        self.write_op(dest, op1, op2, "+")

    def as_string(self):
        self.apply_var_map()
        lines = [
            "// + XOR",
            "// & AND",
            "// Summand a with n bits, i0 = LSB",
            "// Summand b with n bits, in = LSB",
            "",
            "INPUTS {}".format(" ".join(self.inputs)),
            "OUTPUTS {}".format(" ".join(self.outputs)),
            "",
        ]
        if self.flex:
            lines.append(f"CONTROLS s[{self.flex_bits}]")
        for dest, op1, op2, op in self.ops:
            if op == "&" and self.flex:
                lines.append(f"{dest} = fAND[s]({op1}, {op2})")
            else:
                lines.append(f"{dest} = {op1} {op} {op2}")
        lines.append("")
        return "\n".join(lines)

    def loose_vars(self):
        var_loose = dict()
        for dest, op1, op2, _ in self.ops:
            var_loose.setdefault(dest, True)
            var_loose[op1] = False
            var_loose[op2] = False
        for var in self.outputs:
            var_loose[var] = False
        return [var for var, loose in var_loose.items() if loose]


# [RC1] Generates binary adder (ripple carry)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = (a_i & b_i) ^ (b_i & c_i) ^ (a_i &  c_i)
# n = number bits each summand has
# Tested for n = 4 and n = 16
def print_RC1(c, n, ina, inb, out):
    # Bit 0:
    c.xor_op("o0", ina[0], inb[0])
    c.and_op("c1", ina[0], inb[0])
    # Other bits
    for i in range(1, n):
        c.xor_op(f"t0_{i}", ina[i], inb[i])
        c.xor_op(out[i], f"t0_{i}", f"c{i}")
        c.and_op(f"t1_{i}", ina[i], inb[i])
        c.and_op(f"t2_{i}", inb[i], f"c{i}")
        c.and_op(f"t3_{i}", ina[i], f"c{i}")
        c.xor_op(f"t4_{i}", f"t1_{i}", f"t2_{i}")
        if i == (n - 1):
            c.xor_op(out[i + 1], f"t4_{i}", f"t3_{i}")
        else:
            c.xor_op(f"c{i+1}", f"t4_{i}", f"t3_{i}")


# [RC2] Generates binary adder (ripple carry, improved)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = (a_i & b_i) ^ ((a_i ^ b_i) & c_i)
# n = number bits each summand has
def print_RC2(c, n, ina, inb, out):
    # Bit 0:
    c.xor_op("o0", ina[0], inb[0])
    c.and_op("c1", ina[0], inb[0])
    # Other bits
    for i in range(1, n):
        c.xor_op(f"t0_{i}", ina[i], inb[i])
        c.xor_op(out[i], f"t0_{i}", f"c{i}")
        c.and_op(f"t1_{i}", ina[i], inb[i])
        c.and_op(f"t2_{i}", f"t0_{i}", f"c{i}")
        if i == (n - 1):
            c.xor_op(out[i + 1], f"t1_{i}", f"t2_{i}")
        else:
            c.xor_op(f"c{i+1}", f"t1_{i}", f"t2_{i}")


# [RC3] Generates binary adder (ripple carry, more improved)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = a_i ^ ( (a_i ^ b_i) & (a_i ^ c_(i-1) ))
# n = number bits each summand has
def print_RC3(c, n, ina, inb, out):
    # Bit 0:
    c.xor_op("o0", ina[0], inb[0])
    c.and_op("c1", ina[0], inb[0])
    # Other bits
    for i in range(1, n):
        c.xor_op(f"t0_{i}", ina[i], inb[i])
        c.xor_op(out[i], f"t0_{i}", f"c{i}")
        c.xor_op(f"t1_{i}", ina[i], f"c{i}")
        c.and_op(f"t2_{i}", f"t0_{i}", f"t1_{i}")
        if i == (n - 1):
            c.xor_op(out[i + 1], ina[i], f"t2_{i}")
        else:
            c.xor_op(f"c{i+1}", ina[i], f"t2_{i}")


# [KS] Generates binary adder (koggle stone)
# n = number bits each summand has
def print_KS(c, n, ina, inb, out):
    levels = int(np.ceil(np.log2(n)))
    c.assign("P0_0", out[0])
    c.assign(f"G{levels}_{n-1}", out[n])

    # Initialization - level 0
    for i in range(0, n):
        c.xor_op(f"P0_{i}", ina[i], inb[i])
        c.and_op(f"G0_{i}", ina[i], inb[i])

    # Further levels
    t_cnt = 0
    for level in range(1, levels + 1):
        distance = 2 ** (level - 1)
        distance_next = 2**level

        # Green
        for i in range(0, distance):
            c.assign(f"G{level}_{i}", f"G{level-1}_{i}")

        # Orange
        for i in range(distance, n):
            if not (0 <= i < distance_next):
                c.and_op(f"P{level}_{i}", f"P{level-1}_{i}", f"P{level-1}_{i-distance}")

            c.and_op(f"t{t_cnt}", f"P{level-1}_{i}", f"G{level-1}_{i-distance}")
            c.xor_op(f"G{level}_{i}", f"t{t_cnt}", f"G{level-1}_{i}")
            t_cnt += 1

    # Postprocessing
    for i in range(1, n):
        c.xor_op(out[i], f"P0_{i}", f"G{levels}_{i-1}")


# [sklansky] Generates binary adder (sklansky)
# n = number bits each summand has
# (non-optimal, adapt manually)
def print_sklansky(c, n, ina, inb, out):
    levels = int(np.ceil(np.log2(n)))
    c.assign("P0_0", out[0])

    # Initialization - level 0
    for i in range(0, n):
        c.xor_op(f"P0_{i}", ina[i], inb[i])
        c.and_op(f"G0_{i}", ina[i], inb[i])

    # Further levels
    step = 1
    t_cnt = 0
    for level in range(1, levels + 1):
        for i in range(0, n):
            skip = ((i // step)) % 2 == 0
            if skip:
                if i >= (2**level):
                    c.assign(f"P{level}_{i}", f"P{level-1}_{i}")
                c.assign(f"G{level}_{i}", f"G{level-1}_{i}")
            else:
                prev = ((i // step)) * step - 1
                if i >= (2**level):
                    c.and_op(f"P{level}_{i}", f"P{level-1}_{i}", f"P{level-1}_{prev}")
                c.and_op(f"t{t_cnt}", f"P{level-1}_{i}", f"G{level-1}_{prev}")
                c.xor_op(f"G{level}_{i}", f"t{t_cnt}", f"G{level-1}_{i}")
                t_cnt += 1
        step = step * 2

    # Postprocessing
    for i in range(1, n):
        c.xor_op(out[i], f"P0_{i}", f"G{levels}_{i-1}")
    c.assign(out[n], f"G{levels}_{n-1}")


def find_valid_idx(valid, level, start, n):
    for i in range(start, n):
        if valid[level][i]:
            return i


def find_valid_level(valid, bit_nr):
    mrv = -1
    for level in valid:
        if valid[level][bit_nr]:
            mrv = level
    return mrv


def is_valid(valid, level, i):
    return (i >= 0) and valid[level][i]


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# [BK] Generates binary adder (brent-kung) mod (2^n)
# n = number bits each summand has
def print_BK(c, n, ina, inb, out):
    c.assign("P0_0", out[0])
    # valid-map
    valid = {}
    num_levels = int(np.log2(next_power_of_2(n))) * 2
    for j in range(0, num_levels):
        valid[j] = {}
        for i in range(0, n):
            valid[j][i] = None

    # Initialization - level 0
    for i in range(0, n):
        c.xor_op(f"P0_{i}", ina[i], inb[i])
        c.and_op(f"G0_{i}", ina[i], inb[i])
        valid[0][i] = True

    t_cnt = 0

    # Binary tree - merge 2 elements
    depth_binary_tree = num_levels // 2 + 1
    # assign(f"G{depth_binary_tree-1}_{n-1}", f"o{n}")
    for level in range(1, depth_binary_tree):
        prev = find_valid_idx(valid, level - 1, 0, n)
        while prev != None:
            if (prev + 1) == n:
                c.assign(f"P{level}_{prev}", f"P{level-1}_{prev}")
                c.assign(f"G{level}_{prev}", f"G{level-1}_{prev}")
                valid[level][prev] = True
                prev = None
            else:
                inp = find_valid_idx(valid, level - 1, prev + 1, n)
                if not ((n == next_power_of_2(n)) and ((2**level) - 1) == inp):
                    c.and_op(
                        f"P{level}_{inp}",
                        f"P{level-1}_{inp}",
                        f"P{level-1}_{prev}",
                    )
                c.and_op(f"t{t_cnt}", f"P{level-1}_{inp}", f"G{level-1}_{prev}")
                c.xor_op(f"G{level}_{inp}", f"t{t_cnt}", f"G{level-1}_{inp}")
                t_cnt += 1
                valid[level][inp] = True
                prev = find_valid_idx(valid, level - 1, inp + 1, n)

    # Inverse binary tree
    level = depth_binary_tree - 1
    step = 2 ** (level - 2)
    root_idx = next_power_of_2(n) // 2 - 1
    last_valid_level = find_valid_level(valid, root_idx)
    c.assign(f"G{level}_{root_idx}", f"G{last_valid_level}_{root_idx}")
    valid[level][root_idx] = True

    while step != 0:
        level = level + 1
        root_idx += step
        for i in range(root_idx, -1, (-1) * step):
            prev = i - step
            inp = i
            if inp >= n:
                continue
            if is_valid(valid, level - 1, prev):
                last_valid_level = find_valid_level(valid, inp)
                c.and_op(
                    f"t{t_cnt}", f"P{last_valid_level}_{inp}", f"G{level-1}_{prev}"
                )
                c.xor_op(f"G{level}_{inp}", f"t{t_cnt}", f"G{last_valid_level}_{inp}")
                t_cnt += 1
                valid[level][inp] = True
            else:
                last_valid_level = find_valid_level(valid, inp)
                c.assign(f"G{level}_{inp}", f"G{last_valid_level}_{inp}")
                valid[level][inp] = True
        step = step // 2

    # Postprocessing
    for i in range(1, n):
        last_valid_level = find_valid_level(valid, i - 1)
        c.xor_op(out[i], f"P0_{i}", f"G{last_valid_level}_{i-1}")

    last_valid_level = find_valid_level(valid, n - 1)
    c.assign(f"G{last_valid_level}_{n-1}", out[n])


def print_adder(adder_circuit_function, f, n, mod, test, flex):
    # mod = True: compute carry-out, otherwise computation addition mod 2**n
    no = n if mod else n + 1
    ina = [f"i{i}" for i in range(0, n)]
    inb = [f"i{i}" for i in range(n, 2 * n)]
    out = [f"o{i}" for i in range(0, no)]
    c = Circuit(ina + inb, out, flex=flex)
    if mod:
        adder_circuit_function(c, n - 1, ina[:-1], inb[:-1], out[:-1] + ["last_carry"])
        c.xor_op("sum_last", ina[-1], inb[-1])
        c.xor_op(out[-1], "sum_last", "last_carry")
    else:
        adder_circuit_function(c, n, ina, inb, out)
    f.write(c.as_string())

    assert not c.loose_vars(), f"Loose vars: {c.loose_vars()}"

    if test:
        test_circuit(c.as_string(), n, mod)


def test_circuit(s, n, mod, max_iter=10000):
    from . import circuit_eval

    c = circuit_eval.CircuitEval(s)

    def x2bits(x, base, n=n):
        return {f"i{base+i}": bool((x >> i) & 0x1) for i in range(n)}

    if 2 ** (2 * n) <= max_iter:
        test_cases = it.product(range(2**n), range(2**n))
    else:
        test_cases = (
            (random.randrange(2**n), random.randrange(2**n))
            for _ in range(max_iter)
        )

    for a, b in test_cases:
        eval_res = c.evaluate(x2bits(a, 0) | x2bits(b, n))
        no = n if mod else n + 1
        for i in range(no):
            assert (
                bool(((a + b) >> i) & 0x1) == eval_res[f"o{i}"]
            ), f"Fail eval {a}+{b}, bit {i}"


def cli():
    parser = argparse.ArgumentParser(prog="generate_adder_circuit.py")
    parser.add_argument(
        "-n", dest="n", type=int, required=True, help="bit width of summand"
    )
    adder_types = ["RC1", "RC2", "RC3", "KS", "sklansky", "BK"]
    parser.add_argument(
        "--type",
        dest="adder_type",
        choices=adder_types + [f"{x}mod" for x in adder_types],
        required=True,
        help="Select the adder type (RC = ripple carry v1, RC2 = ripple carry v2, RC3 = ripple carry v3, KS = koggle stone, sklansky = sklansky, BK = brent kung)",
    )
    parser.add_argument("--flex", action="store_true", help="Use flex AND gadgets.")
    parser.add_argument(
        "--out", type=Path, required=True, help="Path of the output file."
    )
    parser.add_argument("--test", action="store_true", help="Run a test of the adder.")
    return parser.parse_args()


def main():
    args = cli()
    adder_type = args.adder_type
    mod = adder_type.endswith("mod")
    if mod:
        adder_type = adder_type.removesuffix("mod")
    print_function = globals()[f"print_{adder_type}"]
    with open(args.out, "w") as f:
        print_adder(print_function, f, args.n, mod, args.test, args.flex)


if __name__ == "__main__":
    main()

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

# Main script for COMPRESS. We take as input
# - a description of a circuit
# - a list of gadget descriptions (function, latencies, randomness usage)
# - a list of gadget areas
# - the area cost of a bit of randomness
# - a circuit latency
# and output
# - a verilog module implementing the circuit
# - a verilog header file describing the randomness usage.

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
from enum import Enum
import itertools as it
import json
from math import ceil
from pathlib import Path
import time
from typing import Sequence, Mapping, Tuple, Set, NewType, Callable, Optional, Any
import re
from shutil import rmtree

import cpmpy as cp
import numpy as np
import tomli

##################################
####### Circuit descrtiption #####
##################################

# We work with Boolean circuits where the gates will be turned in gadgets. We
# never explicitly handle the gates inside the gadgets: gadgets are opaque
# primitive elements from the point of view of this script.
# The same holds for variables: they represent non-masked values, or,
# equivalently, complete sharings. We never manipulate individual shares
# directly.

class Operation(str, Enum):
    '''Logic gate in the Boolean circuit.'''
    XOR = '+'
    XNOR = '#'
    AND = '&'
    NOT = '!'
    # Cross-domain part of an AND gate.
    CROSS = 'cross'
    # Inner-domain part of an AND gate.
    INNER = 'inner'
    # Toffoli gate. Let [op0, op1, op2, op3, ...] be the operands, this
    # computes (op0 & op1) ^ op2 ^ op3 ^ ...
    Toffoli = 'toffoli'


# Variable in the Boolean circuit
Variable = NewType('Variable', str)


@dataclass(frozen=True)
class Computation:
    '''A way to compute the value of a variable in the circuit.'''
    operation: Operation
    operands: Sequence[Variable]


@dataclass
class Circuit:
    '''A Boolean circuit.

    It is made of input variables and intermediate variables. There may be
    multiple ways to compute each intermediate variables, which are represented
    by a set of Computation objects (each Computation object has an Operation
    and a list of operands).
    For example, a variable 'z' can be computed as 'z = x & y' or as 'z =
    z-cross ^ z-inner' where 'z-cross = CROSS(x, y)' and 'z-inner = INNER(x,
    y)'.
    '''
    # Invariants:
    # - all inputs are distinct
    # - all inputs and keys of 'computations' are in all_vars
    # - an input cannot appear in computations
    # - all operands of computations belong to all_vars
    # - there cannot be any computational cycle

    inputs: Sequence[Variable] = field(default_factory=list)
    outputs: Sequence[Variable] = field(default_factory=list)
    # Maps a variable to the ways of computing it.
    computations: Mapping[Variable, Set[Computation]] = field(default_factory=dict)
    # Maps a variable to the variables that have it as an operand.
    dependents: Mapping[Variable, Set[Variable]] = field(default_factory=dict)

    @property
    def all_vars(self):
        return set(self.computations.keys()) | set(self.inputs)

    def _add_input(self, input_):
        'Private builder method.'
        assert input_ not in self.all_vars
        self.inputs.append(input_)
        self.dependents[input_] = set()

    def _add_output(self, output):
        assert output not in self.outputs
        assert output in self.all_vars
        self.outputs.append(output)

    def _add_computation(self, res, computation):
        assert all(op in self.all_vars for op in computation.operands)
        assert res not in self.inputs
        self.computations.setdefault(res, set()).add(computation)
        for op in computation.operands:
            self.dependents[op].add(res)
        self.dependents[res] = set()

    @classmethod
    def from_circuit_str(cls, circuit: str):
        '''Convert the file representation of Boolean circuits to a Circuit.'''
        # Example circuit string:
        #     // A comment
        #     INPUTS i0 i1 i2
        #     OUTPUTS o0 o1
        #     o0 = i0 + i1 // a XOR gate
        #     t = i0 # i1 // a XNOR gate
        #     o1 = i0 & t // an AND gate
        c = Circuit()
        # Split into lines, remove comments and extraneous whitespace
        l = [line.split('//')[0].strip() for line in circuit.splitlines()]
        # Remove empty lines
        l = [line for line in l if line]
        # There can be only one input and one output line.
        inputs_line = next(line for line in l if line.startswith('INPUTS'))
        outputs_line = next(line for line in l if line.startswith('OUTPUTS'))
        # Convert inputs and outputs to Variable type.
        inputs = [Variable(x) for x in inputs_line.strip().split()[1:] if x]
        outputs = [Variable(x) for x in outputs_line.strip().split()[1:] if x]
        # Add inputs to the circuit
        for input_ in inputs:
            c._add_input(input_)
        # Process computations
        computation_lines = [line for line in l if line != inputs_line and line != outputs_line]
        for computation in computation_lines:
            if Operation.NOT in computation: # handle NOT case
                beg, op, op1 = computation.partition(Operation.NOT)
                res, eq = beg.split()
                assert eq == '=', (eq, computation)
                c._add_computation(Variable(res), Computation(Operation(op), (Variable(op1), )))
            else:
                res, eq, op1, op, op2 = computation.split()
                assert eq == '=', (eq, computation)
                c._add_computation(Variable(res), Computation(Operation(op), (Variable(op1), Variable(op2))))
        # Add inputs to the circuit
        for output in outputs:
            c._add_output(output)
        return c

    def split_and_inner_cross(self):
        '''Add a new way to compute AND gates.

        Enable more flexibility in the AND gadget implementation: we split them
        into a cross-domain part and a inner-domain part.
        The new computation can be done by the combination of a CROSS, INNER and XOR gate.
        '''
        for res, computations in self.computations.copy().items():
            try:
                computation = next(c for c in computations if c.operation == Operation.AND)
            except StopIteration:
                continue
            cross = Variable(res + '_cross')
            inner = Variable(res + '_inner')
            assert cross not in self.computations
            assert inner not in self.computations
            self.computations[cross] = { Computation(Operation.CROSS, computation.operands) }
            self.computations[inner] = { Computation(Operation.INNER, computation.operands) }
            self.computations[res].add(Computation(Operation.XOR, (cross, inner)))


    def make_toffolis(self):
        '''Add a new way to compute a & b ^ c ^ d...'''
        for and_res, computations in self.computations.copy().items():
            try:
                and_res_computation = next(c for c in computations if c.operation == Operation.AND)
            except StopIteration:
                continue
            # Starting from x = and_res, iteratively follow the operations:
            # if x is used only once in a xor and the other operand is only used once, let x be the result of the xor
            # otherwise, stop
            x = and_res
            if len(self.dependents[x]) > 1:
                continue
            while True:
                if len(self.dependents[x]) != 1:
                    break
                candidate = next(iter(self.dependents[x]))
                computations  = self.computations[candidate]
                if len(computations) != 1:
                    break
                computation = next(iter(computations))
                if computation.operation != Operation.XOR:
                    break
                other_op = next(op for op in computation.operands if op != x)
                if len(self.dependents[other_op]) > 1:
                    break
                x = candidate
            if x == and_res:
                continue
            res = x
            # Do a backward pass from res: identify all the variables that get
            # XORed to form res, eliminating intermediate variables that are used
            # only once.
            xor_list = []
            explore_stack = [*next(iter(self.computations[res])).operands]
            while explore_stack:
                y = explore_stack.pop()
                if len(self.dependents[y]) > 1 or y not in self.computations:
                    xor_list.append(y)
                else:
                    computations = self.computations[y]
                    computation = next(iter(computations))
                    if len(computations) == 1 and computation.operation == Operation.XOR:
                        explore_stack.extend(computation.operands)
                    else:
                        xor_list.append(y)
            xor_list.remove(and_res)
            self.computations[res].add(Computation(Operation.Toffoli, (*and_res_computation.operands, *xor_list)))
        print('n possible toffoli:', len([c for cs in self.computations.values() for c in cs if c.operation == Operation.Toffoli]))


##################################
############ Gadgets #############
##################################

def randomness_req(rnd_gtype):
    match rnd_gtype:
        case 'HPC2':
            return ('d*(d-1)/2', lambda d: d*(d-1)//2)
        case 'HPC3':
            return ('d*(d-1)', lambda d: d*(d-1))
        case _:
            assert False

@dataclass
class GateImpl:
    "A masked gadget."
    name: str
    latencies: Sequence[Mapping[int, str]]
    # Maps port name to (latency, rng cost in bits for selected nshares, rng cost as verilog expression)
    randomness_usage: Mapping[str, Tuple[int, int, str]]
    area_ge: float
    rng_area_ge: float
    @property
    def has_clk(self):
        return any(x != 0 for x in [l for d in self.latencies for l in d]) or any(lat != 0 for lat, _, _ in self.randomness_usage.values())

    @property
    def cost_ge(self):
        # Ignores fixed rnd area
        return self.area_ge + sum([self.rng_area_ge*rng_cost for _, rng_cost, _ in self.randomness_usage.values()])

    @property
    def cost_ge_int(self):
        SCALING = 100
        return round(SCALING * self.cost_ge)


@dataclass(frozen=True)
class Parameters:
    num_shares: int
    latency: int
    circuit: Path
    out: Path
    outh: Path
    toffoli: bool
    gadgets_area_csv: Path
    rng_area_txt: Path
    outstats: Path
    gadgets_config: Path
    time_limit: float # in seconds


##################################
######## SAT-modeling ############
##################################

# Introduce the basic structure for the modelling of our circuit.
# For each variable, and for each pipeline stage, we have:
# - a 'valid' flag: Is the variable available to be used in a computation at
#   that stage?
# - a 'compute' flag: Is there a gadget that outputs that variable at that
#   stage?
# - a 'pipeline' flag: Is there a pipelining register that could propagate
#   that variable from the previous stage?
# We add the constraints to compute 'valid' from 'compute' and 'pipeline'
# variables, as well as the initial 'compute' for inputs and 'valid'
# constraints for outputs.
# Then, it remains only to constrain the 'compute' variables in terms of gadget
# instantiations (to be done elsewhere).
def model_basic(circuit, latency):
    m = cp.Model()
    m.var_valid = {v: [cp.boolvar(name=f'valid_{v}[{i}]') for i in range(latency+1)] for v in circuit.all_vars}
    m.var_pipeline = {v: [cp.boolvar(name=f'pipeline_{v}[{i}]') for i in range(latency)] for v in circuit.all_vars}
    m.var_compute = {v: [cp.boolvar(name=f'compute_{v}[{i}]') for i in range(latency+1)] for v in circuit.all_vars}

    # Valid is "computed at the current cycle, or valid at the previous cycle
    # and there is a pipeline reg.
    for v in circuit.all_vars:
        # No pipeline from the previous stage for the first stage.
        m += (m.var_valid[v][0] == m.var_compute[v][0])
        for i in range(1, latency+1):
            m += (m.var_valid[v][i] == (m.var_compute[v][i] | (m.var_valid[v][i-1] & m.var_pipeline[v][i-1])))

    # Inputs are "computed" at cycle 0, never after.
    for v in circuit.inputs:
        m += m.var_compute[v][0]
        for i in range(1, latency+1):
            m += (~m.var_compute[v][i])

    # Output must be valid at the end.
    for v in circuit.outputs:
        m += m.var_valid[v][latency]

    return m


# List of possible gadget for each gate.
# Currently costs are set based on GE estimates for a 2-share impl. using umc65LL

def parse_synthcsv(num_shares: int, gacsv: Path):
    def intOrNone(x):
        return int(x) if x else None
    areas = {}

    with open(gacsv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            areas[row['design']] = float(row['area_ge'])
    return areas


def gen_gate_impls(num_shares: int, gacsv: Path, rngtxt: Path, gadgets_config: Path):
    areas = parse_synthcsv(num_shares, gacsv)
    gate_impls = {op:[] for op in Operation}
    with open(gadgets_config, 'rb') as f:
        gadgets = tomli.load(f)['gadget']
    with open(rngtxt, 'r') as f:
        m = float(f.read())
    for gadget in gadgets:
        latencies = []
        for port in gadget['port']:
            latencies.append({ port['latency']: port['name'] })
            if port.get('has_prev'):
                latencies[-1][port['latency']-1] = port['name']+'_prev'
        randomness_usage= dict()
        for rngport in gadget.get('random_port', dict()):
            rnd_req_str, rnd_req_fn = randomness_req(rngport['n_bits'])
            randomness_usage[rngport['name']] = (rngport['latency'], rnd_req_fn(num_shares), rnd_req_str)
        gate = GateImpl(
            name=gadget['name'], 
            latencies=latencies,
            randomness_usage=randomness_usage,
            area_ge=areas[gadget['name']],
            rng_area_ge=m,
        )
        gate_impls[Operation[gadget['operation']]].append(gate)

    return gate_impls, areas['MSKreg'], m


# Here we add the variables representing the instantiations of the possible gadgets.
# We also constrain the 'compute' variable as follows: a variable is computed
# if one of the possible gadgets is instantiated, and its inputs are valid.
# Also add variables that correspond to the number of gadgets istantiated.
def model_gadgets(m, circuit, latency, gadget_library, gate_impls):
    # For verilog export
    m.inst_gadgets = []
    # For cost formulas
    inst_gadgets_kind = {g: list() for g in gadget_library}
    toffoli_regs = []
    # For sharing unicity
    inst_gadgets_v = {v: set() for v in circuit.computations.keys()}
    inst_gadgets_vc = {v: defaultdict(set) for v in circuit.computations.keys()}
    m.toffoli_inst = []
    possible_computations = defaultdict(list)
    m.debug_info = defaultdict(dict)
    ## Instantiation of gadgets and "compute" variable assignment.
    for v, computations in circuit.computations.items():
        for l in range(latency+1):
            for i, computation in enumerate(computations):
                for gadget in gate_impls[computation.operation]:
                    if l >= max(op_l for lats in gadget.latencies for op_l in lats):
                        # Whether we instantiate the gadget
                        inst = cp.boolvar(name=f'{gadget.name}-{v}-{i}[{l}]')
                        m.inst_gadgets.append((inst, v, computation, l, gadget.name, dict()))
                        inst_gadgets_kind[gadget.name].append(inst)
                        inst_gadgets_v[v].add(inst)
                        inst_gadgets_vc[v][(computation, gadget.name)].add(inst)
                        operands_valid = cp.all(
                                m.var_valid[op][l-op_l]
                                for op, latencies in zip(computation.operands, gadget.latencies)
                                for op_l in latencies
                                )
                        possible_computations[(l, v)].append(inst & operands_valid)
                if computation.operation == Operation.Toffoli:
                    # Toffoli-capable gates
                    toffoli_gadgets = set(('MSKand_hpc2o2', 'MSKand_hpc2o2_swapped', 'MSKand_hpc3o', 'MSKand_hpc3o_swapped'))
                    toffoli_gadgets = [gadget for gadget in gate_impls[Operation.AND] if gadget.name in toffoli_gadgets]
                    for gadget in toffoli_gadgets:
                        if l >= max(op_l for lats in gadget.latencies for op_l in lats):
                            # Whether we instantiate the gadget
                            inst = cp.boolvar(name=f'toffoli_{gadget.name}-{v}-{i}[{l}]')
                            # Var for the AND gate and operands.
                            m.toffoli_inst.append((inst, v, computation, l))
                            and_operands_valid = cp.all(
                                    m.var_valid[op][l-op_l]
                                    for op, latencies in zip(computation.operands[:2], gadget.latencies)
                                    for op_l in latencies
                                    )
                            # Var for XOR operands
                            xor_list = computation.operands[2:]
                            # Add count of XORs for Toffoli gates
                            xor_gadget = gate_impls[Operation.XOR][0].name
                            late_latencies = list(range(l, latency+1))
                            xor_latencies = [l-1] + late_latencies
                            reg_forward_out = [cp.boolvar(name=f'toffoli_{gadget.name}-{v}-{i}[{l}]_fwd_{j}') for j in range(latency-l)]
                            toffoli_regs.extend(reg_forward_out)
                            if reg_forward_out:
                                m += reg_forward_out[0].implies(inst)
                            for o, o_later in zip(reg_forward_out[:-1], reg_forward_out[1:]):
                                m += o_later.implies(o)
                            xor_ops_valid = [cp.boolvar(name=f'toffoli_{gadget.name}-{v}-{i}[{l}]_xorop-{op}-valid') for op in xor_list]
                            for op, op_valid_val in zip(xor_list, xor_ops_valid):
                                m += (op_valid_val == cp.any(
                                    enable_cond & m.var_valid[op][lat]
                                    for enable_cond, lat in zip([True, True, *reg_forward_out], xor_latencies)
                                    ))
                            # All operands of the XOR list are valid at some point in time.
                            xor_operands_valid = cp.all(xor_ops_valid)
                            # At least one operand comes in early (otherwise the Toffoli gate doesn't help).
                            xor_operands_early = cp.boolvar(name=f'toffoli_{gadget.name}-{v}-{i}[{l}]_anyxorearly')
                            m += (xor_operands_early == cp.any(m.var_valid[op][l-1] for op in xor_list))
                            m.debug_info[(v, l, gadget.name)] = dict(
                                    inst=inst,
                                    and_operands_valid=and_operands_valid,
                                    xor_list=xor_list,
                                    late_latencies=late_latencies,
                                    reg_forward_out=reg_forward_out,
                                    xor_ops_valid = xor_ops_valid,
                                    xor_operands_valid = xor_operands_valid,
                                    xor_operands_early = xor_operands_early,
                                    pcs = dict(),
                                    )
                            m.inst_gadgets.append((inst, v, computation, l, gadget.name, dict(reg_forward_out=reg_forward_out)))
                            inst_gadgets_kind[gadget.name].append(inst)
                            inst_gadgets_v[v].add(inst)
                            inst_gadgets_vc[v][(computation, gadget.name)].add(inst)
                            inst_gadgets_kind[xor_gadget].append(len(xor_list)*inst)
                            for lat in range(l, latency+1):
                                # We forward up to latency l (implies inst).
                                fwd = reg_forward_out[lat-l-1] if lat > l else inst
                                # We don't forward later (otherwise operands from the xor_list could be XORed in later, and we could be invalid).
                                fwd_stop = ~reg_forward_out[lat-l] if lat < latency else True
                                poss_cmp = cp.boolvar(name=f'toffoli_{gadget.name}-{v}-{i}[{l}]_pcmp')
                                m += (poss_cmp == (fwd & fwd_stop & and_operands_valid & xor_operands_valid & xor_operands_early))
                                possible_computations[(lat, v)].append(poss_cmp)
                                m.debug_info[(v, l, gadget.name)]['pcs'][lat] = dict(fwd=fwd, fwd_stop=fwd_stop, poss_cmp=poss_cmp, all_pc=possible_computations)
            m += (m.var_compute[v][l] == cp.any(possible_computations[(l,v)]))

    ## Top-level counts
    m.gadget_count = {gadget: cp.sum(insts) for gadget, insts in inst_gadgets_kind.items()}
    m.n_regs = cp.sum(v for vs in m.var_pipeline.values() for v in vs) + cp.sum(toffoli_regs)
    ## Unicity of var generation in order to ensure equality of all sharings of the same variable.
    def atmostone(variables):
        return cp.sum(variables) <= 1
    # Unicity for randomness-using gadgets: only one instance.
    for v, insts in inst_gadgets_v.items():
        uses_rnd = any(
                bool(gadget.randomness_usage)
                for computation in circuit.computations[v]
                for gadget in gate_impls[computation.operation]
                )
        if uses_rnd:
            m += atmostone(insts)
    # Unicity for all gadgets: only one kind of gadget (may be instantiated multiple times).
    for v, cginsts in inst_gadgets_vc.items():
        m += atmostone(cp.any(insts) for insts in cginsts.values())

# List all the gadgets instantiated by the solve model.
def list_gadgets(m):
    gadget_list = []
    for inst, v, computation, l, gadget_name, opts in m.inst_gadgets:
        if inst.value():
            gadget_list.append((gadget_name, v, l, computation, opts))
    return gadget_list


##################################
###### Verilog generation ########
##################################

# Groups the randomness requirements of the submodules per latency, and
# generate the fragments of verilog code needed to index them.
def rnd_reqs(gadgets, gadget_library):
    randoms = dict()
    gadget_rnd = dict()
    def format_rnd_count(rnd_count):
        if len(rnd_count.items()) == 0:
            return '0'
        else:
            return '+'.join(f'{n}*({count})' for count, n in rnd_count.items())
    for gadget, v, i, _ in gadgets:
        rnds_gadget =  dict()
        for name, (lat, _, count) in gadget_library[gadget].randomness_usage.items():
            global_lat = i-lat
            lat_rnd_count = randoms.setdefault(global_lat, dict())
            rnds_gadget[lat] = f'.{name}(rnd{global_lat}[{format_rnd_count(lat_rnd_count)} +: {count}])'
            lat_rnd_count.setdefault(count, 0)
            lat_rnd_count[count] += 1
        gadget_rnd[(gadget, v, i)] = rnds_gadget
    random_list = [f'rnd{l}' for l in randoms]
    randoms = {l: format_rnd_count(rnd_count) for l, rnd_count in randoms.items()}
    random_decls = [
            f'(* fv_type="random", fv_count=1, fv_rnd_count_0={rnd_count}, fv_rnd_lat_0={l}  *)\n'
            + f'input [{rnd_count}-1:0] rnd{l};'
            for l, rnd_count in randoms.items()
            ]
    return randoms, random_list, random_decls, gadget_rnd

@dataclass
class VerilogGenerator:
    circuit: Circuit
    latency: int
    module_name: str
    num_shares: int
    gadget_library: Mapping[str, GateImpl]
    gadgets: Sequence[tuple[str, str, int, Computation, Sequence[Any]]]
    reg_pipeline: Sequence[tuple[str, int]]
    var_valid: Mapping[str, Sequence[bool]]
    m: Any

    def file_header(self):
        return [
                '`timescale 1ns/1ps\n',
                f'// latency = {self.latency}\n',
                f'// Fully pipeline PINI circuit in {self.latency} clock cycles.',
                '// This file has been automatically generated.',
                '`ifdef FULLVERIF',
                '(* fv_prop = "PINI", fv_strat = "composite", fv_order=d *)',
                '`endif',
                ]

    def random_lat_count_assign(self):
        lat_count = defaultdict(lambda: defaultdict(lambda: 0))
        rnd_assign = dict()
        for gadget, v, i, _, _ in self.gadgets:
            for name, (lat, _, size_expr) in self.gadget_library[gadget].randomness_usage.items():
                global_lat = i - lat
                rnd_assign[(gadget, v, i)] = lat_count[global_lat].copy()
                lat_count[global_lat][size_expr] += 1
        return lat_count, rnd_assign

    def random_lat_count(self):
        return self.random_lat_count_assign()[0]

    def random_assign(self):
        return self.random_lat_count_assign()[1]

    def random_lats(self):
        return list(self.random_lat_count().keys())

    def ports(self):
        return ['clk', *self.circuit.inputs, *self.circuit.outputs, *(f'rnd{l}' for l in self.random_lats())]

    def format_rnd_count(self, rnd_exprs):
        if rnd_exprs:
            return '+'.join(f'{n}*({count})' for count, n in rnd_exprs.items())
        else:
            # nothing seen yet, hence 0
            return '0'

    def rnd_count_exprs(self):
        return sorted((lat, self.format_rnd_count(rnd_exprs)) for lat, rnd_exprs in self.random_lat_count().items())

    def port_decls(self):
        clk_decl =  ('(* fv_type="clock" *)', 'input clk;')
        input_decls = [
                    ('(* fv_type="sharing", fv_latency=0, fv_count=1 *)', f'input [d-1:0] {x};')
                    for x in self.circuit.inputs
                    ]
        output_decls = [
                    (f'(* fv_type="sharing", fv_latency={self.latency}, fv_count=1 *)', f'output [d-1:0] {x};')
                    for x in self.circuit.outputs
                    ]
        rnd_decls = [
                (f'(* fv_type="random", fv_count=1, fv_rnd_count_0={rnd_count}, fv_rnd_lat_0={lat}  *)', f'input [{rnd_count}-1:0] rnd{lat};')
                for lat, rnd_count in self.rnd_count_exprs()
        ]
        return [clk_decl] + input_decls + output_decls + rnd_decls


    def sharings(self):
        res = [
                *[(v, 0) for v in self.circuit.inputs],
                *[(v, i) for _, v, i, _, _ in self.gadgets],
                *[(v, i+1) for v, i in self.reg_pipeline],
                ]
        res.sort()
        return res

    def gadget_instantiation(self, gadget_kind, name, ports, additional_params=None):
        if additional_params is None:
            additional_params = dict()
        additional_params['d'] = 'd'
        param_str =  ', '.join(f'.{pn}({pv})' for pn, pv in additional_params.items())
        if gadget_kind == 'MSKreg' or gadget_kind.endswith('_tof') or self.gadget_library[gadget_kind].has_clk:
            ports['clk'] = 'clk'
        return [
                f'{gadget_kind} #({param_str}) {name} (',
                ',\n'.join(f'    .{pn}({pv})' for pn, pv in ports.items()),
                ');',
                ]

    def generate_random_header(self):
        return '\n'.join([
            '// Randomness bus sizes.',
            '// This file has been automatically generated.',
            *(f'localparam rnd_bus{l} = {rnd};' for l, rnd in self.rnd_count_exprs()),
            '',
            ])

    def generate_top(self):
        lines = []
        # Module header
        lines += self.file_header()
        lines.append(f'module {self.module_name} # ( parameter d={self.num_shares} ) (')
        lines += [f'    {x},' for x in self.ports()]
        # I/Os
        lines.append(');')
        lines.append('`include "MSKand_hpc2.vh"')
        lines.append('`include "MSKand_hpc3.vh"')
        # I/Os wire declaration (old-style).
        for attr, decl in self.port_decls():
            lines.append(attr)
            lines.append(decl)
        # Internal wires
        lines.extend(f'wire [d-1:0] {v}_{i};' for v, i in self.sharings())
        # Assign I/Os
        lines.extend(f'assign {v}_0 = {v};' for v in self.circuit.inputs)
        lines.extend(f'assign {v} = {v}_{self.latency};' for v in self.circuit.outputs)
        lines.append('\n')
        # Gadget instantiation
        rnd_assign = self.random_assign()
        for (gadget, v, l, computation, opts) in self.gadgets:
            ports = dict()
            if computation.operation == Operation.Toffoli:
                # Identify out regs
                fwd_regs = [eval_expr(x) for x in opts['reg_forward_out']]
                n_fwd_regs = len(list(it.takewhile(lambda x: x, fwd_regs)))
                # Instantiate all XORs and additional sharing wires.
                early_ops = []
                late_ops = [[] for _ in range(n_fwd_regs+1)]
                for op in computation.operands[2:]:
                    if self.var_valid[op][l-1]:
                        early_ops.append(op)
                    else:
                        first_lat = next(i for i, x in enumerate(self.var_valid[op][l:]) if x)
                        late_ops[first_lat].append(op)
                early_tmps = [f'{v}_{l}_tof_early_{i}' for i, _ in enumerate(early_ops[:-1])]
                late_tmps = [[f'{v}_{l}_tof_late_{i}_{j}' for j in range(len(ops)+1)] for i, ops in enumerate(late_ops)]
                for x in it.chain(early_tmps, *late_tmps):
                    lines.append(f'wire [d-1:0] {x};')
                early_ops_fmt = [f'{early_op}_{l-1}' for early_op in early_ops]
                for i, tmp in enumerate(early_tmps):
                    lines.extend(self.gadget_instantiation(
                        'MSKxor',
                        f'comp_{v}_{l}_tof_xor_early_{i}',
                        {'ina': early_tmps[i-1] if i != 0 else early_ops_fmt[0], 'inb': early_ops_fmt[i+1], 'out': tmp}
                        ))
                for i, ops in enumerate(late_ops):
                    for j, op in enumerate(ops):
                        assert self.var_valid[op][l+i], f'Toffoli for {v}, {l} ({gadget}): {op} not valid at {l+i}.'
                        lines.extend(self.gadget_instantiation(
                            'MSKxor',
                            f'comp_{v}_{l}_tof_xor_late_{i}_{j}',
                            {'ina': late_tmps[i][j], 'inb': f'{op}_{l+i}', 'out': late_tmps[i][j+1]}
                            ))
                    if i != 0:
                        lines.extend(self.gadget_instantiation(
                            'MSKreg',
                            f'comp_{v}_{l}_tof_reg_late_{i}',
                            {'in': late_tmps[i-1][-1], 'out': late_tmps[i][0]}
                            ))
                lines.append(f'assign {v}_{l} = {late_tmps[-1][-1]};')
                # particular cases for the Toffoli gadget.
                gadget_name = f'{gadget}_tof'
                ports['inc'] = early_tmps[-1] if early_tmps else early_ops_fmt[0]
                ports['out'] = late_tmps[0][0]
            else: # Non-Toffoli gadget
                gadget_name = gadget
                ports['out'] = f'{v}_{l}'
            ports.update(
                    (name, f'rnd{l-lat}[{self.format_rnd_count(rnd_assign[(gadget, v, l)])} +: {count}]')
                    for name, (lat, _, count) in self.gadget_library[gadget].randomness_usage.items()
                    )
            operands_ports = computation.operands[:2] if computation.operation == Operation.Toffoli else computation.operands
            for op, lats in zip(operands_ports, self.gadget_library[gadget].latencies):
                ports.update((n, f'{op}_{l-l_op}') for l_op, n in lats.items())
            lines.extend(self.gadget_instantiation(gadget_name, f'comp_{v}_{l}', ports))
        # MSKreg instances
        for v, i in self.reg_pipeline:
            lines.extend(self.gadget_instantiation('MSKreg', f'reg_{v}_{i}', {'in': f'{v}_{i}', 'out': f'{v}_{i+1}'}))
        lines.append('endmodule')
        return '\n'.join(lines) + '\n'


def generate_verilog(circuit, latency, module_name, m, gadget_library, num_shares):
    reg_pipeline = [(v, i) for v, insts in m.var_pipeline.items() for i, inst in enumerate(insts) if inst.value()]
    var_valid = {op: [eval_expr(vv) for vv in var_valid] for op, var_valid in m.var_valid.items()}
    gen = VerilogGenerator(circuit, latency, module_name, num_shares, gadget_library, list_gadgets(m), reg_pipeline, var_valid, m)
    return gen.generate_top(), gen.generate_random_header()


# Workaround, since some expressions need a .value() to evaluate, while
# others are numpy arrays (cpmpy weirdness).
def eval_expr(x):
    try:
        return int(x)
    except TypeError:
        return x.value()

##################################
###### Top-level functions #######
##################################


def generate_masked_circuit(params: Parameters):
    with open(params.circuit, 'r') as f:
        s = f.read()
    c = Circuit.from_circuit_str(s)
    if params.toffoli:
        c.make_toffolis()
    c.split_and_inner_cross()

    m = model_basic(c, params.latency)
    gate_impl, reg_cost, rng_cost_per_bit = gen_gate_impls(params.num_shares, params.gadgets_area_csv, params.rng_area_txt, params.gadgets_config)
    gadget_library = {gadget.name: gadget for gadgets in gate_impl.values() for gadget in gadgets}
    model_gadgets(m, c, params.latency, gadget_library, gate_impl)
    m.minimize(cp.sum(ceil(gadget_library[gadget].cost_ge_int) * count for gadget, count in m.gadget_count.items()) + reg_cost * m.n_regs)

    t0 = time.time()
    hassol = m.solve(time_limit=params.time_limit)
    solve_time = time.time() - t0
    print("Status:", m.status())  # Status: ExitStatus.OPTIMAL (0.03033301 seconds)
    if hassol:
        print("solution found.")
        ge_sum = 0
        rng_sum = 0
        stats = {}
        for gadget in gadget_library:
            print(f'{gadget}: {eval_expr(m.gadget_count[gadget])}')
            stats[gadget] = int(eval_expr(m.gadget_count[gadget]))
            ge_sum += eval_expr(m.gadget_count[gadget]) * gadget_library[gadget].cost_ge
            rng_sum += eval_expr(m.gadget_count[gadget]) * gadget_library[gadget].randomness_usage.get('rnd', [0,0])[1]
        print(f'MSKreg: {eval_expr(m.n_regs)}')
        ge_sum += reg_cost * eval_expr(m.n_regs)
        print(f'Approx. area cost (GE): {float(ge_sum)}')
        print(f'#RNG bits {rng_sum}')
        print('Toffoli:',  len([x for x, *_ in m.toffoli_inst if eval_expr(x)]))
        stats["MSKreg"] = eval_expr(m.n_regs)
        stats["Area Estimate (GE)"] = float(ge_sum)
        stats["RNG Bits"] = int(rng_sum)
        stats["Toffoli"] = len([x for x, *_ in m.toffoli_inst if eval_expr(x)])
        stats["Num Shares"] = params.num_shares
        stats["Latency"] = params.latency
        stats["Use Toffoli"] = params.toffoli
        stats["Circuit"] = str(params.circuit)
        stats["RNG Cost Per Bit"] = rng_cost_per_bit
        stats["solve_time"] = solve_time
        module_name = params.circuit.stem
        verilog, verilog_header = generate_verilog(c, params.latency, module_name, m, gadget_library, params.num_shares)
        params.out.parent.mkdir(exist_ok=True, parents=True)
        with open(params.out, 'w') as f:
            f.write(verilog)
        if params.outh is not None:
            params.outh.parent.mkdir(exist_ok=True, parents=True)
            with open(params.outh, 'w') as f:
                f.write(verilog_header)
        with open(params.outstats, 'w') as f:
            json.dump(stats, f, indent=4)
    else:
        print("UNSAT.")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True, type=Path, help="Input circuit file.", )
    parser.add_argument('--latency', '-l',metavar="CLK CYCLES", type=int, default=6, help='Desired latency, default 4')
    parser.add_argument('--out', required=True, type=Path, help='Generated verilog file.')
    parser.add_argument('--outh', type=Path, help='Generated verilog header file.')
    parser.add_argument('--num-shares', '-d', default=2, metavar="D", type=int, help='protection order, default 2')
    parser.add_argument('--toffoli', action=argparse.BooleanOptionalAction, default=True, help="Use Toffoli gates in the circuit.")
    parser.add_argument('--gadgets-area-csv',type=Path, required=True)
    parser.add_argument('--rng-area-txt',type=Path, required=True)
    parser.add_argument('--outstats', required=True, type=Path)
    parser.add_argument('--gadgets-config', required=True)
    parser.add_argument('--time-limit', type=float, default=300.0, help="Solver timeout in second.")
    return parser


def main():
    args = cli().parse_args()
    assert args.circuit.is_file(), f"Circuit file {args.circuit} does not exist."
    gp = Parameters(**vars(args))
    generate_masked_circuit(gp)


if __name__ == '__main__':
    main()

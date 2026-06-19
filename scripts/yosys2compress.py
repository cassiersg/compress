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

"""
Conversion of Yosys output JSON for a combinational circuit into a .txt file
usable as a COMPRESS input.
"""

import argparse
import json
import networkx as nx
import dataclasses


class VarNameGen:
    def __init__(self):
        self.value = 0

    def gen(self):
        res = f"gen{self.value}"
        self.value += 1
        return res


@dataclasses.dataclass(frozen=True, eq=True)
class DepNode:
    outputs: tuple[str, ...]
    inputs: tuple[str, ...]
    template: str

    def render(self) -> str:
        return self.template.format(*self.outputs, *self.inputs)


def sorted_stmts(nodes) -> list[str]:
    G = nx.DiGraph()
    output_map = dict()
    for node in nodes:
        G.add_node(node)
        for v in node.outputs:
            if v not in output_map:
                output_map[v] = node
            else:
                raise ValueError(f"Re-defining variable {v}.")
    primary_inputs = set()
    for node in nodes:
        for input in node.inputs:
            if input not in output_map and input not in primary_inputs:
                primary_inputs.add(input)
                output_map[input] = input
                G.add_node(input)
            G.add_edge(output_map[input], node)
    return [n.render() for n in nx.topological_sort(G) if n not in primary_inputs]


def fetch_variable_name(varmap, bit_index, autogen_index):
    name_var = varmap.get(bit_index)
    if name_var is None:
        name_var = autogen_index.gen()
        varmap[bit_index] = name_var
    return name_var


COMB_CELLS = {
    "$_NOT_": ([("Y", 0)], [("A", 0)], "{} = !{}"),
    "$_XOR_": ([("Y", 0)], [("A", 0), ("B", 0)], "{} = {} + {}"),
    "$_AND_": ([("Y", 0)], [("A", 0), ("B", 0)], "{} = {} & {}"),
    "$_MUX_": ([("Y", 0)], [("S", 0), ("A", 0), ("B", 0)], "{} = MUX2[{}]({},{})"),
    "G4_mul": (
        [("z", i) for i in range(2)],
        [(v, i) for v in ["x", "y"] for i in range(2)],
        "({}, {}) = G4_mul({}, {}, {}, {})",
    ),
    "G16_mul": (
        [("z", i) for i in range(4)],
        [(v, i) for v in ["x", "y"] for i in range(4)],
        "({}, {}, {}, {}) = G16_mul({}, {}, {}, {}, {}, {}, {}, {})",
    ),
}


def process_instance(cell_name, cell_inst, varmap, index):
    inst_type = cell_inst["type"]
    conns = cell_inst["connections"]
    if inst_type in COMB_CELLS:
        inputs, outputs, template = COMB_CELLS[inst_type]
        depnode = DepNode(
            tuple(fetch_variable_name(varmap, conns[x][i], index) for x, i in inputs),
            tuple(fetch_variable_name(varmap, conns[x][i], index) for x, i in outputs),
            template,
        )
        ctrls = conns["S"] if inst_type == "$_MUX_" else []
        return depnode, ctrls
    else:
        raise ValueError(f"Cell type {inst_type} unknown ({cell_name}, {cell_inst})")


def port_definitions(topmod, varmap, controls) -> list[str]:
    def pdecl(pname, ports):
        return (pname + " " + " ".join(varmap[b] for b in ports)).strip()

    inputs = [
        b
        for port in topmod["ports"].values()
        for b in port["bits"]
        if port["direction"] == "input" and b not in controls
    ]
    outputs = [
        b
        for port in topmod["ports"].values()
        for b in port["bits"]
        if port["direction"] == "output"
    ]
    return [
        pdecl("INPUTS", inputs),
        pdecl("OUTPUTS", outputs),
        pdecl("CONTROLS", controls),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yosys to COMPRESS input formatter")
    parser.add_argument(
        "--netlist-file",
        type=str,
        default="yosys_canright_aes_sbox_trivial_from_c.json",
        help="Yosys JSON netlist file",
    )
    parser.add_argument("--top", type=str, default="Sbox", help="")
    parser.add_argument(
        "--compress-file",
        type=str,
        default="canright-compress.txt",
        help="COMPRESS input file",
    )
    args = parser.parse_args()

    with open(args.netlist_file, "r") as f:
        yosys_netlist = json.load(f)

    top_module = yosys_netlist["modules"][args.top]
    # initial set of variables (top-level inputs and outputs)
    for port in top_module["ports"].values():
        assert port["direction"] in ("input", "output")
    variables_map = {
        x: "{}{}".format(pname, i)
        for pname, port in top_module["ports"].items()
        for i, x in enumerate(port["bits"])
    }

    # Process all instances in the top module
    index_cnt = VarNameGen()
    depnodes, ctrls = zip(
        *[
            process_instance(instance, cell_inst, variables_map, index_cnt)
            for instance, cell_inst in yosys_netlist["modules"][args.top][
                "cells"
            ].items()
        ]
    )

    # Port definitions
    ctrls = set(c for cs in ctrls for c in cs)
    ports_defs = port_definitions(top_module, variables_map, ctrls)
    # Generate assgnments for all variables (except inputs)
    stmts = sorted_stmts(depnodes)

    with open(args.compress_file, "w") as f:
        f.write("\n".join(ports_defs + stmts))

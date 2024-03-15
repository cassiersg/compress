// SPDX-FileCopyrightText: SIMPLE-Crypto Contributors <info@simple-crypto.dev>
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright SIMPLE-Crypto Contributors.
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
// You may redistribute and modify this source and make products using it under
// the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).
// This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
// OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
// Please see the CERN-OHL-P v2 for applicable conditions.

// Masked AND HPC2 gadget with swapped inputs.
`ifdef FULLVERIF
(* fv_strat = "flatten", fv_order=d *)
`endif
`ifndef DEFAULTSHARES
`define DEFAULTSHARES 2
`endif
`ifndef SHIDX_BITS
    // Lower-bound on the number of bits of d-1. Taking 3 as a default: works
    // for all d <= 8.
    `define SHIDX_BITS 3
`endif
module fMSKand_hpc2o_swapped #(parameter d=`DEFAULTSHARES) (ina, ina_prev, inb, rnd, clk, out, s);

`include "MSKand_hpc2.vh"

(* fv_type = "sharing", fv_latency = 0 *) input  [d-1:0] ina;
(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] ina_prev;
(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] inb;
(* fv_type = "random", fv_count = 1, fv_rnd_lat_0 = 0, fv_rnd_count_0 = hpc2rnd *) input [hpc2rnd-1:0] rnd;
(* fv_type = "clock" *) input clk;
(* fv_type = "sharing", fv_latency = 2 *) output [d-1:0] out;
(* fv_type = "control", fv_latency = 2 *) input [`SHIDX_BITS-1:0] s;

fMSKand_hpc2o #(.d(d)) inner(
    .ina(inb),
    .inb(ina),
    .inb_prev(ina_prev),
    .rnd(rnd),
    .clk(clk),
    .out(out),
    .s(s)
);

endmodule

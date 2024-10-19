// SPDX-FileCopyrightText: SIMPLE-Crypto Contributors <info@simple-crypto.dev>
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright SIMPLE-Crypto Contributors.
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
// You may redistribute and modify this source and make products using it under
// the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).
// This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
// OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
// Please see the CERN-OHL-P v2 for applicable conditions.

// Masked HPC1 G(16) multiplication.
`ifdef FULLVERIF
(* fv_prop = "PINI", fv_strat = "assumed", fv_order=d *)
`endif
`ifdef MATCHI
(* matchi_prop = "PINI", matchi_strat = "assumed", matchi_shares=d, matchi_arch="pipeline" *)
`endif
`ifndef DEFAULTSHARES
`define DEFAULTSHARES 2
`endif
module MSKg16mul_hpc1 #(parameter integer d=`DEFAULTSHARES)
(
    ina0,
    ina1,
    ina2,
    ina3,
    inb0,
    inb1,
    inb2,
    inb3,
    rnd_ref,
    rnd_mul,
    clk,
    out0,
    out1,
    out2,
    out3
);
`include "MSKand_hpc1.vh"

(* matchi_type = "sharing", matchi_latency = 1+ref_rndlat *)
(* fv_type = "sharing", fv_latency = 1+ref_rndlat *)
input  [d-1:0] ina0, ina1, ina2, ina3;
(* matchi_type = "sharing", matchi_latency = ref_rndlat *)
(* fv_type = "sharing", fv_latency = ref_rndlat *)
input  [d-1:0] inb0, inb1, inb2, inb3;
(* matchi_type = "clock", fv_type = "clock" *)
input clk;
(* matchi_type = "sharing", matchi_latency = 2+ref_rndlat *)
(* fv_type = "sharing", fv_latency = 2+ref_rndlat *)
output [d-1:0] out0, out1, out2, out3;
(* matchi_type = "random", matchi_latency = 0 *)
(* fv_type = "random", fv_count=1, fv_rnd_lat_0=0, fv_rnd_count_0=4*ref_n_rnd *)
input [4*ref_n_rnd-1:0] rnd_ref;
(* matchi_type = "random", matchi_latency = 1+ref_rndlat *)
(* fv_type = "random", fv_count=1, fv_rnd_lat_0=1+ref_rndlat, fv_rnd_count_0=4*dom_rnd *)
input [4*dom_rnd-1:0] rnd_mul;

wire [d-1:0] inb0_ref, inb1_ref, inb2_ref, inb3_ref;

MSKref_sni #(.d(d))
rfrsh0 (.in(inb0), .clk(clk), .out(inb0_ref), .rnd(rnd_ref[0 +: ref_n_rnd]));

MSKref_sni #(.d(d))
rfrsh1 (.in(inb1), .clk(clk), .out(inb1_ref), .rnd(rnd_ref[ref_n_rnd +: ref_n_rnd]));

MSKref_sni #(.d(d))
rfrsh2 (.in(inb2), .clk(clk), .out(inb2_ref), .rnd(rnd_ref[2*ref_n_rnd +: ref_n_rnd]));

MSKref_sni #(.d(d))
rfrsh3 (.in(inb3), .clk(clk), .out(inb3_ref), .rnd(rnd_ref[3*ref_n_rnd +: ref_n_rnd]));

MSKg16mul_dom #(.d(d)) mul (
    .ina0(ina0),
    .ina1(ina1),
    .ina2(ina2),
    .ina3(ina3),
    .inb0(inb0_ref),
    .inb1(inb1_ref),
    .inb2(inb2_ref),
    .inb3(inb3_ref),
    .clk(clk),
    .rnd(rnd_mul),
    .out0(out0),
    .out1(out1),
    .out2(out2),
    .out3(out3)
);

endmodule

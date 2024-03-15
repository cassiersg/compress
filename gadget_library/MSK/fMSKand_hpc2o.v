// SPDX-FileCopyrightText: SIMPLE-Crypto Contributors <info@simple-crypto.dev>
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright SIMPLE-Crypto Contributors.
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
// You may redistribute and modify this source and make products using it under
// the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).
// This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
// OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
// Please see the CERN-OHL-P v2 for applicable conditions.

// Masked AND HPC2 gadget (only cross-domain terms).
`ifdef FULLVERIF
(* fv_prop = "PINI", fv_strat = "assumed", fv_order=d *)
`endif
`ifndef DEFAULTSHARES
`define DEFAULTSHARES 2
`endif
`ifndef SHIDX_BITS
    // Lower-bound on the number of bits of d-1. Taking 3 as a default: works
    // for all d <= 8.
    `define SHIDX_BITS 3
`endif
module fMSKand_hpc2o #(parameter d=`DEFAULTSHARES) (ina, inb, inb_prev, rnd, clk, out, s);

`include "MSKand_hpc2.vh"

(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] ina;
(* fv_type = "sharing", fv_latency = 0 *) input  [d-1:0] inb;
(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] inb_prev;
(* fv_type = "random", fv_count = 1, fv_rnd_lat_0 = 0, fv_rnd_count_0 = hpc2rnd *) input [hpc2rnd-1:0] rnd;
(* fv_type = "clock" *) input clk;
(* fv_type = "random", fv_type = "sharing", fv_latency = 2 *) output [d-1:0] out;
(* fv_type = "control", fv_latency = 2 *) input [`SHIDX_BITS-1:0] s;
                                      
genvar i,j;

// unpack vector to matrix --> easier for randomness handling
//reg [hpc2rnd-1:0] rnd_prev;
wire [hpc2rnd-1:0] rnd_prev;
bin_REG #(.W(hpc2rnd)) REGin_rnd_prev (
    .clk(clk),
    .in(rnd),
    .out(rnd_prev)
);

wire [d-1:0] rnd_mat [d-1:0]; 
wire [d-1:0] rnd_mat_prev [d-1:0]; 
for(i=0; i<d; i=i+1) begin: igen
    assign rnd_mat[i][i] = 0;
    assign rnd_mat_prev[i][i] = 0;
    for(j=i+1; j<d; j=j+1) begin: jgen
        assign rnd_mat[j][i] = rnd[((i*d)-i*(i+1)/2)+(j-1-i)];
        // The next line is equivalent to
        //assign rnd_mat[i][j] = rnd_mat[j][i];
        // but we changed it for Verilator efficient simulation -> Avoid UNOPFLAT Warning (x2 simulation perfs enabled)
        assign rnd_mat[i][j] = rnd[((i*d)-i*(i+1)/2)+(j-1-i)];
        assign rnd_mat_prev[j][i] = rnd_prev[((i*d)-i*(i+1)/2)+(j-1-i)];
        // The next line is equivalent to
        //assign rnd_mat_prev[i][j] = rnd_mat_prev[j][i];
        // but we changed it for Verilator efficient simulation -> Avoid UNOPFLAT Warning (x2 simulation perfs enabled)
        assign rnd_mat_prev[i][j] = rnd_prev[((i*d)-i*(i+1)/2)+(j-1-i)];
    end
end

for(i=0; i<d; i=i+1) begin: ParProdI
    wire [d-2:0] uw;
    assign out[i] = ^uw;
    for(j=0; j<d; j=j+1) begin: ParProdJ
        wire [`SHIDX_BITS-1:0] off_diag = ~(i^j);
        wire enable = |(off_diag & s);
        if (i != j) begin: NotEq
            localparam j2 = j < i ?  j : j-1;
            wire u_comb;
            // j2 == 0: u = Reg(not(a_i)*r_ij + a_i*b_i)
            // j2 != 0: u = Reg(not(a_i)*r_ij)
            if (j2 != 0) begin
                assign u_comb = ~ina[i] & rnd_mat_prev[i][j];
            end else begin
                assign u_comb = (~ina[i] & rnd_mat_prev[i][j]) ^ (ina[i] & inb_prev[i]);
            end
            wire u;
            bin_REG #(.W(1)) REGin_u(
                .clk(clk),
                .in(u_comb),
                .out(u)
            );
            // w = Reg[a_i * Reg(b_j + r_ij)]
            wire v_comb = inb[j] ^ rnd_mat[i][j];
            wire v;
            bin_REG #(.W(1)) REGin_v(
                .clk(clk),
                .in(v_comb),
                .out(v)
            );
            wire w_comb = ina[i] & v;
            wire w;
            bin_REG #(.W(1)) REGin_w(
                .clk(clk),
                .in(w_comb),
                .out(w)
            );
            assign uw[j2] = enable & (u ^ w);
        end
    end
end

endmodule

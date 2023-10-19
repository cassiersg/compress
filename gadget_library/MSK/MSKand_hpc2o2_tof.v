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
module MSKand_hpc2o2_tof #(parameter d=`DEFAULTSHARES) (ina, inb, inb_prev, inc, rnd, clk, out);

`include "MSKand_hpc2.vh"

(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] ina;
(* fv_type = "sharing", fv_latency = 0 *) input  [d-1:0] inb;
(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] inc;
(* fv_type = "sharing", fv_latency = 1 *) input  [d-1:0] inb_prev;
(* fv_type = "random", fv_count = 1, fv_rnd_lat_0 = 0, fv_rnd_count_0 = hpc2rnd *) input [hpc2rnd-1:0] rnd;
(* fv_type = "clock" *) input clk;
(* fv_type = "random", fv_type = "sharing", fv_latency = 2 *) output [d-1:0] out;
                                      
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

wire [d-1:0] not_ina;
bin_NOT #(.W(d)) NOTin_not_ina (
    .in(ina),
    .out(not_ina)
);
for(i=0; i<d; i=i+1) begin: ParProdI
    wire [d-2:0] u, v, w, uw;
    
    wire ru_xor_rw;
    bin_redXOR #(.W(d-1)) redXORin_red_uw(
        .in(uw),
        .out(ru_xor_rw)
    );
    assign out[i] = ru_xor_rw;
    for(j=0; j<d; j=j+1) begin: ParProdJ
        if (i != j) begin: NotEq
            localparam j2 = j < i ?  j : j-1;
            wire u_j2_comb;
            // j2 == 0: u[j2] = Reg(not(a_i)*r_ij + a_i*b_i)
            // j2 != 0: u[j2] = Reg(not(a_i)*r_ij)
            if (j2 != 0) begin
                bin_AND #(.W(1)) ANDin_u_j2_comb(
                    .ina(not_ina[i]),
                    .inb(rnd_mat_prev[i][j]),
                    .out(u_j2_comb)
                );
                bin_OR #(.W(1)) ORuw(
                    .ina(u[j2]),
                    .inb(w[j2]),
                    .out(uw[j2])
                );
            end else begin
                wire nai_rij, aibi, aibici;
                
                bin_AND #(.W(1)) ANDnai_rij(
                    .ina(not_ina[i]),
                    .inb(rnd_mat_prev[i][j]),
                    .out(nai_rij)
                );
                bin_AND #(.W(1)) ANDaibi(
                    .ina(ina[i]),
                    .inb(inb_prev[i]),
                    .out(aibi)
                );
                bin_XOR #(.W(1)) XORtoffoli(
                    .ina(aibi),
                    .inb(inc[i]),
                    .out(aibici)
                );
                bin_XOR #(.W(1)) XORin_u_j2_comb (
                    .ina(nai_rij),
                    .inb(aibici),
                    .out(u_j2_comb)
                );
                bin_XOR #(.W(1)) XORuw(
                    .ina(u[j2]),
                    .inb(w[j2]),
                    .out(uw[j2])
                );
            end
            bin_REG #(.W(1)) REGin_u(
                .clk(clk),
                .in(u_j2_comb),
                .out(u[j2])
            );
            // w[j2] = Reg[a_i * Reg(b_j + r_ij)]
            wire v_j2_comb;
            bin_XOR #(.W(1)) XORin_v2_comb(
                .ina(inb[j]),
                .inb(rnd_mat[i][j]),
                .out(v_j2_comb)
            );
            bin_REG #(.W(1)) REGin_v(
                .clk(clk),
                .in(v_j2_comb),
                .out(v[j2])
            );
            wire w_j2_comb;
            bin_AND #(.W(1)) ANDin_w_j2_comb(
                .ina(ina[i]),
                .inb(v[j2]),
                .out(w_j2_comb)
            );
            bin_REG #(.W(1)) REGin_w(
                .clk(clk),
                .in(w_j2_comb),
                .out(w[j2])
            );
        end
    end
end

endmodule

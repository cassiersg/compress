// SPDX-FileCopyrightText: SIMPLE-Crypto Contributors <info@simple-crypto.dev>
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright SIMPLE-Crypto Contributors.
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
// You may redistribute and modify this source and make products using it under
// the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).
// This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
// OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
// Please see the CERN-OHL-P v2 for applicable conditions.

// SNI refresh gadget, for d=2,...,16
`ifdef FULLVERIF
(* fv_prop = "SNI", fv_strat = "assumed", fv_order=d *)
`endif
`ifndef DEFAULTSHARES
`define DEFAULTSHARES 2
`endif
module MSKref_sni #(parameter d=`DEFAULTSHARES) (in, clk, out, rnd);

`include "MSKref_sni.vh"

(* syn_keep="true", keep="true", fv_type="sharing", fv_latency=1 *) input [d-1:0] in;
(* syn_keep="true", keep="true", fv_type="sharing", fv_latency=2 *) output reg [d-1:0] out;
(* fv_type="clock" *) input clk;
(* syn_keep="true", keep="true", fv_type= "random", fv_count=1, fv_rnd_lat_0 = 0, fv_rnd_count_0 = ref_n_rnd *)
input [ref_n_rnd-1:0] rnd;

reg [d-1:0] share0;
always @(posedge clk)
    out <= in ^ share0;

if (d == 1) begin
    always @(posedge clk) begin
        share0 <= 1'b0;
    end
end else if (d == 2) begin
    always @(posedge clk) begin
        share0 <= {rnd[0], rnd[0]};
    end
end else if (d==3) begin
    always @(posedge clk) begin
        share0 <= {rnd[0]^rnd[1], rnd[1], rnd[0]};
    end
end else if (d==4 || d==5) begin
    wire [d-1:0] r1 = rnd[d-1:0];
    always @(posedge clk) begin
        share0 <= r1[d-1:0] ^ { r1[d-2:0], r1[d-1] };
    end
end else if (d <= 12) begin
    wire [d-1:0] r1 = rnd[d-1:0];
    reg [ref_n_rnd-d-1:0] r2;
    always @(posedge clk) r2 <= rnd[ref_n_rnd-1:d];
    reg [d-1:0] s1;
    always @(posedge clk)
        s1 <= r1[d-1:0] ^ { r1[d-2:0], r1[d-1] };
    case (d)
        6: always @(posedge clk)
            share0 <= {s1[d-1:4], s1[3]^r2[0], s1[2:1], s1[0]^r2[0]};
        7: always @(posedge clk)
        share0 <= {
            s1[6]^r2[0],
            s1[5],
            s1[4]^r2[1],
            s1[3],
            s1[2]^r2[0],
            s1[1],
            s1[0]^r2[1]
        };
        8: always @(posedge clk)
        share0 <= {
            s1[7],
            s1[6]^r2[0],
            s1[5]^r2[1],
            s1[4]^r2[2],
            s1[3],
            s1[2]^r2[0],
            s1[1]^r2[1],
            s1[0]^r2[2]
        };
        9: always @(posedge clk)
        share0 <= {
            s1[8],
            s1[7]^r2[0],
            s1[6]^r2[1],
            s1[5],
            s1[4]^r2[2],
            s1[3]^r2[0],
            s1[2],
            s1[1]^r2[1],
            s1[0]^r2[2]
        };
        10: always @(posedge clk)
        share0 <= {
            s1[9]^r2[4],
            s1[8]^r2[3],
            s1[7]^r2[2],
            s1[6]^r2[1],
            s1[5]^r2[0],
            s1[4]^r2[4],
            s1[3]^r2[3],
            s1[2]^r2[2],
            s1[1]^r2[1],
            s1[0]^r2[0]
        };
        11: always @(posedge clk)
        share0 <= {
            s1[10]^r2[0],
            s1[9]^r2[1],
            s1[8]^r2[2],
            s1[7]^r2[3]^r2[0],
            s1[6]^r2[4],
            s1[5]^r2[5],
            s1[4]^r2[1],
            s1[3]^r2[2],
            s1[2]^r2[3],
            s1[1]^r2[4],
            s1[0]^r2[5]
        };
        12: always @(posedge clk)
        share0 <= {
            s1[11]^r2[2]^r2[0],
            s1[10]^r2[3],
            s1[9]^r2[4],
            s1[8]^r2[5]^r2[0],
            s1[7]^r2[6],
            s1[6]^r2[7],
            s1[5]^r2[2]^r2[1],
            s1[4]^r2[3],
            s1[3]^r2[4],
            s1[2]^r2[5]^r2[1],
            s1[1]^r2[6],
            s1[0]^r2[7]
        };
    endcase
end else if (d <= 16) begin
    wire [d-1:0] r1 = rnd[d-1:0];
    wire [ref_n_rnd-d-1:0] r2 = rnd[ref_n_rnd-1:d];
    reg [d-1:0] s1, s2;
    always @(posedge clk) begin
        s1 <= r1[d-1:0] ^ { r1[d-2:0], r1[d-1] };
        s2 <= r1[d-1:0] ^ { r2[d-4:0], r2[d-1:d-3] };
        share0 <= s1 ^ s2;
    end
end

endmodule

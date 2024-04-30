// SPDX-FileCopyrightText: SIMPLE-Crypto Contributors <info@simple-crypto.dev>
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright SIMPLE-Crypto Contributors.
// This source describes Open Hardware and is licensed under the CERN-OHL-P v2.
// You may redistribute and modify this source and make products using it under
// the terms of the CERN-OHL-P v2 (https://ohwr.org/cern_ohl_p_v2.txt).
// This source is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING
// OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
// Please see the CERN-OHL-P v2 for applicable conditions.

// multiplication in GF(2^2) using normal basis (Omega^2, Omega)
module G4_mul(input wire [1:0] x, input wire [1:0] y, output wire [1:0] z);
    wire a, b, c, d, e;
    assign a = x[1];
    assign b = x[0];
    assign c = y[1];
    assign d = y[0];
    assign e = (a ^ b) & (c ^ d);
    assign z[1] = (a & c) ^ e;
    assign z[0] = (b & d) ^ e;
endmodule




// multiplication in GF(2^4) using normal basis (alpha^8, alpha^2)
module G16_mul(input wire [3:0] x, input wire [3:0] y, output wire [3:0] z);
    wire [1:0] a, b, c, d, e, e_scl, p, q;
    assign a = x[3:2];
    assign b = x[1:0];
    assign c = y[3:2];
    assign d = y[1:0];
    G4_mul mul1(.x(a ^ b), .y(c ^ d), .z(e));
    G4_scl_N scl_N(.x(e), .z(e_scl));
    G4_mul mul2(.x(a), .y(c), .z(p));
    G4_mul mul3(.x(b), .y(d), .z(q));
    assign z[3:2] = p ^ e_scl;
    assign z[1:0] = q ^ e_scl;
endmodule

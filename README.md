# COMPRESS

This repository contains the tool COMPRESS described in the paper
["Compress: Reducing Area and Latency of Masked Pipelined Circuits"](https://eprint.iacr.org/2023/1600)

Canonical location: <https://github.com/cassiersg/compress>.

## Dependencies

COMPRESS itself is a python3 script (tested with python3.10), with dependencies given in `requirements.txt`.

Other parts of the flow use iverilog (tested with version 0.11) and yosys
(tested with version 0.33) respectively for simulation and area usage
estimation of the netlist generated by COMPRESS.


## Usage

Example:
```
make CIRCUIT=circuits/bp_sbox.txt "LATS=4 5" "DS=2 3" area -j
```
produces area results for the Boyar-Peralta AES S-box (`circuits/aes_bp.txt`)
with 2 and 3 shares, latency 4 and 5 in `work/aes_bp_area.csv`.

Variables for `make` invocation:
- `WORK`: directory where all intermediate files are stored
- `CIRCUIT`: path to a circuit file (e.g., `circuits/bp_sbox.txt`)
- `LATS`: space-separated list of latencies (a circuit is generated for each latency)
- `DS`: space-separated number of shares (a circuit is generated for each number of shares)
- `GADGETS_CONFIG`: list of gadgets that can be used for synthesis (default: `gadget_library/all_gadgets.toml`)

In order to only run the COMPRESS design generation, use directly
`scripts/compress.py` (run `python3 scripts/compress.py --help` for
command-line parameters).

## Adder generation

The script `scripts/generate_adder_circuit.py` has been used to generate the adder circuits in `circuits/`.

Usage:
```
python3 scripts/generate_adder_circuit.py -n <bit_width> --type <TYPE> --out <OUTPATH>
```
where
* `-n <bit_width>` refers to the input bits of the summands
* `--out` indicates the generated file path
* `--type` refers to the adder type. The following adders are currently supported:
  * `RC1`: standard ripple-carry adder computing the carry as `c_(i+1) = (a_i & b_i) ^ (b_i & c_i) ^ (a_i &  c_i)`
  * `RC2`: slightly improved ripple-carry adder computing the carry as `c_(i+1) = (a_i & b_i) ^ ((a_i ^ b_i) & c_i)`
  * `RC3`: more improved ripple-carry adder computing the carry as `c_(i+1) = a_i ^ ( (a_i ^ b_i) & (a_i ^ c_(i-1) ))`
  * `KS`: Koggle-Stone adder as described in https://www.mdpi.com/2076-3417/12/5/2274
  * `sklansky`: Sklansky adder as described in https://www.mdpi.com/2076-3417/12/5/2274
  * `BK`: Brent-Kung adder as described in https://www.mdpi.com/2076-3417/12/5/2274


## Contents

Selected files/directories:

```
├── circuits # textual description of some circuits to mask
├── gadget_library
│   ├── BIN/bin_REG.v # simple reg, instantiated in masked gadgets
│   ├── *.toml # sets of gadgets with metadata.
│   ├── MSK
│   │   └── *.v, *.vh # implementation of the gadgets
│   └── RNG # PRNG based on Trivum for generating masking randomness.
├── Makefile # Top-level entry point.
├── scripts # python scripts: COMPRESS, its dependencies, pre- and post-processing.
├── simu.mk # behavioral and structural simulation of circuits generated by COMPRESS.
└── synthesis # synthesis with Yosys and Nangate45
```



## License

GPLv3 and CERN-OHL-S v2, see LICENSE.txt

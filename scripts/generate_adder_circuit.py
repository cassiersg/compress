import argparse
from pathlib import Path

import numpy as np

# [RC1] Generates binary adder (ripple carry)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = (a_i & b_i) ^ (b_i & c_i) ^ (a_i &  c_i)
# n = number bits each summand has
# Tested for n = 4 and n = 16
def print_RC1(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    # Adding two n-bit summands will give (n+1)-bit sum
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")

    f.write("\n")

    # Bit 0:
    f.write(f"o0 = i0 + i{n}\n")

    f.write("\n")
    f.write(f"c1 = i0 & i{n}\n")

    f.write("\n")

    for i in range(1, n):
    
        f.write(f"t0_{i} = i{i} + i{n+i}\n")
        f.write(f"o{i} = t0_{i} + c{i}\n")
        
        f.write("\n")

        f.write(f"t1_{i} = i{i} & i{n+i}\n")
        f.write(f"t2_{i} = i{n+i} & c{i}\n")
        f.write(f"t3_{i} = i{i} & c{i}\n")
        f.write(f"t4_{i} = t1_{i} + t2_{i}\n")

        if i == (n-1):
            f.write(f"o{i+1} = t4_{i} + t3_{i}\n")
        else:
            f.write(f"c{i+1} = t4_{i} + t3_{i}\n")

        f.write("\n")


# [RC2] Generates binary adder (ripple carry, improved)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = (a_i & b_i) ^ ((a_i ^ b_i) & c_i)
# n = number bits each summand has
# Tested for n = 4
def print_RC2(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    # Bit 0:
    f.write(f"o0 = i0 + i{n}\n")

    f.write("\n")
    f.write(f"c1 = i0 & i{n}\n")

    f.write("\n")

    for i in range(1, n):
    
        f.write(f"t0_{i} = i{i} + i{n+i}\n")
        f.write(f"o{i} = t0_{i} + c{i}\n")
        
        f.write("\n")

        f.write(f"t1_{i} = i{i} & i{n+i}\n")
        f.write(f"t2_{i} = t0_{i} & c{i}\n")

        if i == (n-1):
            f.write(f"o{i+1} = t1_{i} + t2_{i}\n")
        else:
            f.write(f"c{i+1} = t1_{i} + t2_{i}\n")

        f.write("\n")


# [RC3] Generates binary adder (ripple carry, more improved)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = a_i ^ ( (a_i ^ b_i) & (a_i ^ c_(i-1) ))
# n = number bits each summand has
# Tested for n = 4
def print_RC3(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    # Bit 0:
    f.write(f"o0 = i0 + i{n}\n")

    f.write("\n")
    f.write(f"c1 = i0 & i{n}\n")

    f.write("\n")

    for i in range(1, n):
    
        f.write(f"t0_{i} = i{i} + i{n+i}\n")
        f.write(f"o{i} = t0_{i} + c{i}\n")
        
        f.write("\n")

        f.write(f"t1_{i} = i{i} + c{i}\n")
        f.write(f"t2_{i} = t0_{i} & t1_{i}\n")

        if i == (n-1):
            f.write(f"o{i+1} = i{i} + t2_{i}\n")
        else:
            f.write(f"c{i+1} = i{i} + t2_{i}\n")

        f.write("\n")


# [RC3] Generates binary adder (ripple carry, more improved) mod (2^n)
# sum_i = a_i ^ b_i ^ c_i
# c_(i+1) = a_i ^ ( (a_i ^ b_i) & (a_i ^ c_(i-1) ))
# n = number bits each summand has
def print_RC3mod(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    # Bit 0:
    f.write(f"o0 = i0 + i{n}\n")

    f.write("\n")
    f.write(f"c1 = i0 & i{n}\n")

    f.write("\n")

    for i in range(1, n):
    
        f.write(f"t0_{i} = i{i} + i{n+i}\n")
        f.write(f"o{i} = t0_{i} + c{i}\n")
        
        f.write("\n")

        if i != (n-1):
            f.write(f"t1_{i} = i{i} + c{i}\n")
            f.write(f"t2_{i} = t0_{i} & t1_{i}\n")
            f.write(f"c{i+1} = i{i} + t2_{i}\n")

        f.write("\n")


# [KS] Generates binary adder (koggle stone)
# n = number bits each summand has
def print_KS(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    levels = int(np.ceil(np.log2(n)))

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")

    # Further levels
    t_cnt = 0
    for level in range(1, levels+1):
        distance = 2**(level-1)
        distance_next = 2**level

        # Green
        for i in range(0, distance):
            f.write(f"G{level}_{i} = G{level-1}_{i}\n")

        # Orange
        for i in range(distance, n):
            
            if not(0 <= i < distance_next):
                f.write(f"P{level}_{i} = P{level-1}_{i} & P{level-1}_{i-distance}\n")

            f.write(f"t{t_cnt} = P{level-1}_{i} & G{level-1}_{i-distance}\n")
            f.write(f"G{level}_{i} = t{t_cnt} + G{level-1}_{i}\n")
            t_cnt+=1
    
    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        f.write(f"o{i} = P0_{i} + G{levels}_{i-1}\n")
    f.write(f"o{n} = G{levels}_{n-1}\n")


# [KSmod] Generates binary adder (koggle stone)  mod (2^n)
# n = number bits each summand has
# (non-optimal, adapt manually)
def print_KSmod(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    levels = int(np.ceil(np.log2(n)))

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")

    # Further levels
    t_cnt = 0
    for level in range(1, levels+1):
        distance = 2**(level-1)
        distance_next = 2**level

        # Green
        for i in range(0, distance):
            f.write(f"G{level}_{i} = G{level-1}_{i}\n")

        # Orange
        for i in range(distance, n):
            
            if not(0 <= i < distance_next):
                f.write(f"P{level}_{i} = P{level-1}_{i} & P{level-1}_{i-distance}\n")

            f.write(f"t{t_cnt} = P{level-1}_{i} & G{level-1}_{i-distance}\n")
            f.write(f"G{level}_{i} = t{t_cnt} + G{level-1}_{i}\n")
            t_cnt+=1
    
    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        f.write(f"o{i} = P0_{i} + G{levels}_{i-1}\n")


# [sklansky] Generates binary adder (sklansky)   mod (2^n)
# n = number bits each summand has
# (non-optimal, adapt manually)
def print_sklanskymod(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    levels = int(np.ceil(np.log2(n)))

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")

    # Further levels
    step = 1
    t_cnt = 0
    for level in range(1, levels+1):
        for i in range(0,n):
            skip = ((i // step)) % 2 == 0
            if skip:
                if i >= (2**level):
                    f.write(f"P{level}_{i} = P{level-1}_{i}\n")
                f.write(f"G{level}_{i} = G{level-1}_{i}\n")
            else:
                prev = ((i // step)) * step - 1
                if i >= (2**level):
                    f.write(f"P{level}_{i} = P{level-1}_{i} & P{level-1}_{prev}\n")
                f.write(f"t{t_cnt} = P{level-1}_{i} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{i} = t{t_cnt} + G{level-1}_{i}\n")
                t_cnt+=1
        step = step * 2
    
    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        f.write(f"o{i} = P0_{i} + G{levels}_{i-1}\n")
    #f.write(f"o{n} = G{levels}_{n-1}\n")


# [sklansky] Generates binary adder (sklansky)
# n = number bits each summand has
def print_sklansky(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")

    levels = int(np.ceil(np.log2(n)))

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")

    # Further levels
    step = 1
    t_cnt = 0
    for level in range(1, levels+1):
        for i in range(0,n):
            skip = ((i // step)) % 2 == 0
            if skip:
                if i >= (2**level):
                    f.write(f"P{level}_{i} = P{level-1}_{i}\n")
                f.write(f"G{level}_{i} = G{level-1}_{i}\n")
            else:
                prev = ((i // step)) * step - 1
                if i >= (2**level):
                    f.write(f"P{level}_{i} = P{level-1}_{i} & P{level-1}_{prev}\n")
                f.write(f"t{t_cnt} = P{level-1}_{i} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{i} = t{t_cnt} + G{level-1}_{i}\n")
                t_cnt+=1
        step = step * 2
    
    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        f.write(f"o{i} = P0_{i} + G{levels}_{i-1}\n")
    f.write(f"o{n} = G{levels}_{n-1}\n")


def find_valid_idx(valid, level, start, n):
    for i in range(start, n):
        if valid[level][i]:
            return i
    return None

def find_valid_level(valid, bit_nr):
    mrv = -1
    for level in valid:
        if valid[level][bit_nr]:
            mrv = level
    return mrv

def is_valid(valid, level, i):
    return (i >= 0) and valid[level][i]

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


# [BK] Generates binary adder (brent-kung)
# n = number bits each summand has
def print_BK(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n+1)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")


    # valid-map
    valid = {}
    num_levels = int(np.log2(next_power_of_2(n))) * 2
    for j in range(0, num_levels):
        valid[j] = {}
        for i in range(0, n):
            valid[j][i] = None

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")
        valid[0][i] = True

    t_cnt = 0

    # Binary tree - merge 2 elements
    depth_binary_tree = num_levels//2 + 1
    for level in range(1, depth_binary_tree):
        prev = find_valid_idx(valid, level-1, 0, n)
        while prev != None:
            if (prev+1) == n: 
                f.write(f"P{level}_{prev} = P{level-1}_{prev}\n")
                f.write(f"G{level}_{prev} = G{level-1}_{prev}\n")
                valid[level][prev] = True
                prev = None
            else: 
                inp = find_valid_idx(valid, level-1, prev+1, n)
                if not((n == next_power_of_2(n)) and ((2**level)-1) == inp):
                    f.write(f"P{level}_{inp} = P{level-1}_{inp} & P{level-1}_{prev}\n")
                f.write(f"t{t_cnt} = P{level-1}_{inp} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{inp} = t{t_cnt} + G{level-1}_{inp}\n")
                t_cnt+=1
                valid[level][inp] = True
                prev = find_valid_idx(valid, level-1, inp+1, n)

    # Inverse binary tree
    step = 2**(level-2)
    root_idx = next_power_of_2(n)//2 - 1
    last_valid_level = find_valid_level(valid, root_idx)
    f.write(f"G{level}_{root_idx} = G{last_valid_level}_{root_idx}\n")
    valid[level][root_idx] = True

    while step != 0:
        level = level +1
        root_idx += step
        for i in range(root_idx, -1, (-1) * step):
            prev = i-step
            inp = i
            if inp >= n:
                continue
            if is_valid(valid, level-1, prev):
                last_valid_level = find_valid_level(valid, inp)
                f.write(f"t{t_cnt} = P{last_valid_level}_{inp} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{inp} = t{t_cnt} + G{last_valid_level}_{inp}\n")
                t_cnt+=1
                valid[level][inp] = True
            else:
                last_valid_level = find_valid_level(valid, inp)
                f.write(f"G{level}_{inp} = G{last_valid_level}_{inp}\n")
                valid[level][inp] = True
        step = step // 2

    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        last_valid_level = find_valid_level(valid, i-1)
        f.write(f"o{i} = P0_{i} + G{last_valid_level}_{i-1}\n")
    last_valid_level = find_valid_level(valid, n-1)
    f.write(f"o{n} = G{last_valid_level}_{n-1}\n")

 
# [BK] Generates binary adder (brent-kung) mod (2^n)
# n = number bits each summand has
def print_BKmod(n, f):
    f.write("// + XOR\n")
    f.write("// & AND\n")
    f.write("// Summand a with n bits, i0 = LSB\n")
    f.write("// Summand b with n bits, in = LSB\n")
    f.write("\n")

    f.write("INPUTS ")
    inputs_str = " ".join([f"i{i}" for i in range(0, 2*n)]) 
    f.write(inputs_str)
    f.write("\n")

    f.write("OUTPUTS ")
    outputs_str = " ".join([f"o{i}" for i in range(0, n)])
    f.write(outputs_str)
    f.write("\n")
    f.write("\n")


    # valid-map
    valid = {}
    num_levels = int(np.log2(next_power_of_2(n))) * 2
    for j in range(0, num_levels):
        valid[j] = {}
        for i in range(0, n):
            valid[j][i] = None

    # Initialization - level 0
    for i in range(0, n):
        f.write(f"P0_{i} = i{i} + i{n+i}\n")
        f.write(f"G0_{i} = i{i} & i{n+i}\n")
        valid[0][i] = True

    t_cnt = 0

    # Binary tree - merge 2 elements
    depth_binary_tree = num_levels//2 + 1
    for level in range(1, depth_binary_tree):
        prev = find_valid_idx(valid, level-1, 0, n)
        while prev != None:
            if (prev+1) == n: 
                f.write(f"P{level}_{prev} = P{level-1}_{prev}\n")
                f.write(f"G{level}_{prev} = G{level-1}_{prev}\n")
                valid[level][prev] = True
                prev = None
            else: 
                inp = find_valid_idx(valid, level-1, prev+1, n)
                if not((n == next_power_of_2(n)) and ((2**level)-1) == inp):
                    f.write(f"P{level}_{inp} = P{level-1}_{inp} & P{level-1}_{prev}\n")
                f.write(f"t{t_cnt} = P{level-1}_{inp} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{inp} = t{t_cnt} + G{level-1}_{inp}\n")
                t_cnt+=1
                valid[level][inp] = True
                prev = find_valid_idx(valid, level-1, inp+1, n)

    # Inverse binary tree
    step = 2**(level-2)
    root_idx = next_power_of_2(n)//2 - 1
    last_valid_level = find_valid_level(valid, root_idx)
    f.write(f"G{level}_{root_idx} = G{last_valid_level}_{root_idx}\n")
    valid[level][root_idx] = True

    while step != 0:
        level = level +1
        root_idx += step
        for i in range(root_idx, -1, (-1) * step):
            prev = i-step
            inp = i
            if inp >= n:
                continue
            if is_valid(valid, level-1, prev):
                last_valid_level = find_valid_level(valid, inp)
                f.write(f"t{t_cnt} = P{last_valid_level}_{inp} & G{level-1}_{prev}\n")
                f.write(f"G{level}_{inp} = t{t_cnt} + G{last_valid_level}_{inp}\n")
                t_cnt+=1
                valid[level][inp] = True
            else:
                last_valid_level = find_valid_level(valid, inp)
                f.write(f"G{level}_{inp} = G{last_valid_level}_{inp}\n")
                valid[level][inp] = True
        step = step // 2

    # Postprocessing
    f.write(f"o0 = P0_0\n")
    for i in range(1, n):
        last_valid_level = find_valid_level(valid, i-1)
        f.write(f"o{i} = P0_{i} + G{last_valid_level}_{i-1}\n")
    #last_valid_level = find_valid_level(valid, n-1)
    #f.write(f"o{n} = G{last_valid_level}_{n-1}\n")


def cli():
    parser = argparse.ArgumentParser(prog='generate_adder_circuit.py')
    parser.add_argument('-n', dest='n', type=int, required=True, help='bit width of summand')
    parser.add_argument(
            "--type",
            dest="adder_type",
            choices=["RC1", "RC2", "RC3", "RC3mod", "KS", "KSmod", "sklansky", "sklanskymod", "BK", "BKmod"],
            required=True,
            help="Select the adder type (RC = ripple carry v1, RC2 = ripple carry v2, RC3 = ripple carry v3, KS = koggle stone, sklansky = sklansky, BK = brent kung)"
            )
    parser.add_argument('--out', type=Path, required=True, help='Path of the output file.')
    return parser.parse_args()


PRINT_FUNCTIONS = dict(
        RC1=print_RC1,
        RC2=print_RC2,
        RC3=print_RC3,
        RC3mod=print_RC3mod,
        KS=print_KS,
        KSmod=print_KSmod,
        sklansky=print_sklansky,
        sklanskymod=print_sklanskymod,
        BK=print_BK,
        BKmod=print_BKmod,
        )


def main():
    args = cli()
    with open(args.out, 'w') as f:
        PRINT_FUNCTIONS[args.adder_type](args.n, f)

if __name__ == '__main__':
    main()

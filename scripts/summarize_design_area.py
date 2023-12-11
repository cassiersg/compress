import argparse
import csv
import json
from pathlib import Path

GE = 0.798 # Nangate45


def parse_reports(reports):
    for report in reports:
        with open(report) as f:
            l = json.load(f)
        case_dir = report.parent.parent
        with open(case_dir / 'stats.json') as f:
            stats = json.load(f)
        with open(case_dir.parent.parent / 'rng_area' / 'area_ge.txt') as f:
            rng_area_ge_per_bit = float(f.read())
        module, nshares, _ = str(report.parent.parent.name).rsplit('_', 2)
        nshares = int(nshares[1:])
        area_ge = float(l['design']['area'])/GE
        latency = stats["Latency"]
        if 'RNG Bits' in stats:
            rnd_bits = int(stats['RNG Bits'])
        else:
            rnd_bits = nshares*(nshares-1)//2 * (stats.get('RNG hpc2', 0) + 2 * stats.get('RNG hpc3', 0))
        area_rng = rng_area_ge_per_bit * rnd_bits
        solve_time = stats.get('solve_time', '')
        yield {
                'design': module,
                'nshares': nshares,
                'latency': latency,
                'area_ge': area_ge,
                'area_ge_wrng': area_ge+area_rng,
                'solve_time': solve_time,
                'rnd_bits': rnd_bits,
            }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcsv")
    parser.add_argument("reports", type=Path, metavar='report', nargs='+')
    return parser


def main():
    args = cli().parse_args()
    res = list(parse_reports(args.reports))
    with open(args.outcsv, 'w') as csvfile:
        fields=['design', 'latency', 'nshares', 'area_ge', 'area_ge_wrng', 'solve_time', 'rnd_bits']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(res)
    

if __name__ == "__main__":
    main()

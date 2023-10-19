import argparse
import json
from pathlib import Path

GE = 0.798 # Nangate45


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("reports", type=Path, metavar='report', nargs=2)
    return parser


def main():
    args = cli().parse_args()
    res = []
    for report in args.reports:
        with open(report) as f:
            l = json.load(f)
        nbits = int(str(report.parent.name).split('_', 1)[1])
        areaum2 = float(l["design"].get("area", 0))
        res.append((nbits, areaum2/GE))
    (n1, a1), (n2, a2) = res
    cost = (a2-a1)/(n2-n1)
    with open(args.out, 'w') as f:
        f.write(str(cost))


if __name__ == "__main__":
    main()

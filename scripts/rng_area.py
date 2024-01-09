import argparse
import json
from pathlib import Path

GE = 0.798  # Nangate45


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("report", type=Path)
    return parser


def main():
    args = cli().parse_args()
    with open(args.report) as f:
        l = json.load(f)
    nbits = int(str(args.report.parent.name).split("_", 1)[1])
    areaum2 = float(l["design"].get("area", 0))
    cost = (areaum2 / GE) / nbits
    with open(args.out, "w") as f:
        f.write(str(cost))


if __name__ == "__main__":
    main()

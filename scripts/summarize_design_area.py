import argparse
import csv
import json
from pathlib import Path

GE = 0.798 # Nangate45


def parse_reports(reports):
    for report in reports:
        with open(report) as f:
            l = json.load(f)
        module, nshares, latency = str(report.parent.parent.name).rsplit('_', 2)
        areaum2 = float(l['design']['area'])
        yield {'design': module, 'nshares': nshares, 'latency': latency, 'area_ge': areaum2/GE}


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcsv")
    parser.add_argument("reports", type=Path, metavar='report', nargs='+')
    return parser


def main():
    args = cli().parse_args()
    res = list(parse_reports(args.reports))
    with open(args.outcsv, 'w') as csvfile:
        fields=['design', 'latency', 'nshares', 'area_ge']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(res)
    

if __name__ == "__main__":
    main()

import pandas as pd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('parse_smatch_summary.py')
    parser.add_argument('file', help='file to parse.')
    parser.add_argument('--out_file', help='file to parse.', default="")
    args = parser.parse_args()
    out_file = args.out_file
    if args.out_file == "": 
        out_file=args.file+".csv"
        print(f"--out_file is not passed, setting output to {out_file} instead")
    summary_list = []

    with open(args.file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            name, metric = line.rstrip().split("->")
            P, R, F = map(lambda x: 100*float(x.strip().split(":")[-1]),metric.strip().split(","))
            summary = {}
            summary["Name"] = name.strip()
            summary["P"] = P
            summary["R"] = R
            summary["F"] = F
            summary_list.append(summary)
    pd.DataFrame(summary_list).round(2).to_csv(out_file)
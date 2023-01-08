from main import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('f_in', type=str, help='Input path for file')
parser.add_argument('f_out', type=str, help='Output path for the score file')
args = parser.parse_args()

with open(args.f_in) as in_f, open(args.f_out, 'a') as out_f:
    s = in_f.readlines()

    for line in s:
        scores = None
        f1, f2 = line.split()

        # do 'calculus' with files f1, f2 (compare them)
        score = main(f1, f2)
        out_f.write(f"{score}\n")

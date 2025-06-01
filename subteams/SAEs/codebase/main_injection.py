import os
import argparse
import gzip
import pickle

from src import *

def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations for ODEFormer")
    
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output file/folder")

    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for processing solutions")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing")
    
    parser.add_argument("--inject", nargs="*", type=str, default=[], help="List of places to inject the SAE")
    
    return parser.parse_args()

def main():
    args = parse_args()
    install_sae()
    set_random_seed(8)
    
    inject_fn = inject_and_pass if args.num_threads == 1 else inject_and_pass

    inject_fn(
        args.input, args.output, 
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        to_inject=args.inject
    )

if __name__ == "__main__":
    main()

# Sample command line execution:
# python main_injection.py -i ./input/two_systems_solutions.pkl -o ./output/two_systems/out.pkl.gz \
# --inject encoder.ffns.1 encoder.outer.residual0
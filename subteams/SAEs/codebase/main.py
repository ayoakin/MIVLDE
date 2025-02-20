import os
import argparse
import gzip
import pickle

from src.utils import set_random_seed
from src.instrumentation import install
from src.Keys import Keys
from src.collector import collect

_to_collect = Keys().to_collect.keys()

def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations for ODEFormer")
    
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input file")
    # parser.add_argument("-o", "--output", type=str, required=True, help="Path to output file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")

    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for processing solutions")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing")
    
    parser.add_argument("--encoders", nargs="*", type=int, default=[], help="List of encoder layers to keep")
    parser.add_argument("--decoders", nargs="*", type=int, default=[], help="List of decoder layers to keep")
    
    parser.add_argument("--cross_attention", action="store_true", help="Enable cross attention activations")
    
    parser.add_argument("--encoder_attn", nargs="*", choices={'attn_scores', 'attn_probs', 'attn_output'}, 
                        default=set(), help="Types of encoder attention activations to collect")
    
    for flag in _to_collect:
        parser.add_argument(f"--{flag}", action="store_true", help=f"Enable {flag} collection")

    parser.add_argument("--print", "-p", action="store_true", help=f"View names of collected activation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    install()
    set_random_seed(8)
    
    to_collect = {key: getattr(args, key) for key in _to_collect}
    
    collect(
        args.input, args.output, 
        keys=Keys(
            encoders=set(args.encoders),
            decoders=set(args.decoders),
            encoder_attn=set(args.encoder_attn),
            cross_attention=args.cross_attention,
            to_collect=to_collect
        ),
        num_threads=args.num_threads,
        batch_size=args.batch_size
    )

    if args.print:
        with gzip.open(os.path.join(args.output, 'batch_1.pkl.gz'), 'rb') as file:
            collected_activations = pickle.loads(file.read())

        if not collected_activations:
            return 

        for key, value in collected_activations[0][1].items():
            print(f'{key}')
            for subkey in value:
                print(f'\t{subkey}')

if __name__ == "__main__":
    main()

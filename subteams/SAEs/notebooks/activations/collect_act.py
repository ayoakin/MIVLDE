from collection_utils import collect, Keys
import numpy as np
np.infty = np.inf

def main():
    keys_instance = Keys(
        encoders=[0,1,2,3],
        decoders=[],
        cross_attention=False,
        to_collect={
            'residual_stream': True,
            'mlp_output': True
        }
    )

    collect(
        '../data/solutions/two_systems_solutions.pkl',
        '../data/activations/two_systems_solutions_activations_20k.pkl.gz',
        num_workers=6,
        keys_instance=keys_instance
    )

if __name__ == '__main__':
    main()
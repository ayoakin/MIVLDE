import sys
import uuid
import json
import os
import numpy as np
import time
from tqdm import tqdm
import random
import torch

try:
    from odeformer.envs.environment import FunctionEnvironment
    from odeformer.envs.generators import NodeList
    from parsers import get_parser
except ModuleNotFoundError as e:
    print("[ERROR] Could not import odeformer. Check path and installation.")
    raise e


parser = get_parser()
params = parser.parse_args(args=[])
env = FunctionEnvironment(params)

def identify_operators(sample):
    operators_real = {
      "add": 2, "sub": 2, "mul": 2, "div": 2, "abs": 1, "inv": 1, "sqrt": 1,
      "log": 1, "exp": 1, "sin": 1, "arcsin": 1, "cos": 1, "arccos": 1,
      "tan": 1, "arctan": 1, "pow2": 1, "pow3": 1, 'id': 1
    }
    operators_extra = {"pow": 2}
    all_operators = {**operators_real, **operators_extra}

    skeleton_tree_encoded = sample['skeleton_tree_encoded']
    operator_dict = {operator: 1 if operator in skeleton_tree_encoded else 0 for operator in all_operators}
    sample['operator_dict'] = operator_dict
    return sample

def identify_features(sample):

    def identify_multiple_features(feature_operators, feature_name, operator_dict):
        if any(operator_dict.get(operator, 0) == 1 for operator in feature_operators):
            feature_dict[feature_name] = 1
        else:
            feature_dict[feature_name] = 0

    trig_funs = ['sin', 'cos', 'tan']
    inv_trig_funs = ['arcsin', 'arccos', 'arctan']
    features_single = ['pow2', 'pow3', 'log', 'sqrt', 'exp']

    
    operator_dict = sample['operator_dict']
    feature_dict = {}

    for feature in features_single:
      feature_dict[feature] = operator_dict[feature]

    identify_multiple_features(trig_funs, 'trig', operator_dict)
    identify_multiple_features(inv_trig_funs, 'inv_trig', operator_dict)

    sample['feature_dict'] = feature_dict
    return sample

def generate_samples(save_dir, n_samples):

  os.makedirs(save_dir, exist_ok=True)
  all_samples = {}

  for i in tqdm(range(n_samples)):
    #print('creating sample ', i)
    #tqdm.write(f"Generating sample {i}")
    seed = int(uuid.uuid4().int % (2**32))
    # seed = random.randint(0, 2**32)
    env.rng = np.random.RandomState(seed)
    #tqdm.write(f"seed: {seed}")
    #sample, errors = env.gen_expr(train=False)
    sample = torch.load('.\MIVLDE\subteams\LogitLens\data_gen\sample_249467210.pt')
    #tqdm.write(f"Identifying operators for sample {i}")
    sample = identify_operators(sample)
    #tqdm.write(f"Identifying features for sample {i}")
    sample = identify_features(sample)

    # # check if this sample has any of the features: vast majority seem not to have any
    # feature_dict = sample['feature_dict']
    # if any(value == 1 for value in feature_dict.values()):
    #   print('tree: ', sample['skeleton_tree_encoded'])
    #   print('features: ', sample['feature_dict'])

    #tqdm.write(f"Saving sample {i}")
    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}.json"
    filepath = os.path.join(save_dir, filename)

    # Convert numpy arrays to lists, to be json-compatible
    for key, value in sample.items():
      if isinstance(value, np.ndarray):
        sample[key] = value.tolist()
      
    # remove NodeList entries (i.e. trees) as they are not json-compatible and anyway we have the tree_encoded as lists
    node_keys = [key for key, value in sample.items() if isinstance(value, NodeList)]
    for key in node_keys:
      del sample[key]

    #Save sample to a file
    #with open(filepath, "w") as f:
    #    json.dump(sample, f)

    all_samples[filename] = sample

  # save all samples to a single file
  all_samples_filepath = os.path.join(save_dir, str(n_samples)+"_samples.json")
  with open(all_samples_filepath, "w") as f:
      json.dump(all_samples, f)


  # for sample in samples:
  #     print('skeleton_tree_encoded:', sample['skeleton_tree_encoded'])

  print("[INFO] Data generation complete.")

if __name__ == "__main__":
    n_samples = 1000
    start = time.time()
    generate_samples("./MIVLDE/subteams/LogitLens/data_gen/", n_samples)
    end = time.time()
    print(f"Time taken: {end-start} seconds")
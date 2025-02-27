import os
import numpy as np
import pickle
try:
    from odeformer.envs.environment import FunctionEnvironment
    from parsers import get_parser
except ModuleNotFoundError as e:
    print("[ERROR] Could not import odeformer. Check path and installation.")
    raise e

class RandomSamplesGenerator():
  def __init__(self, samples_path='/content/drive/MyDrive/aisc/samples', num_samples=10, seed=None): # TODO: add samples path
    parser = get_parser()
    params = parser.parse_args(args=[])
    self.env = FunctionEnvironment(params)
    self.samples_path = samples_path
    os.makedirs(self.samples_path, exist_ok=True)
    self.num_samples = num_samples
    self.seed = seed

  def identify_operators(self, sample):
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

  def identify_multiple_features(self, feature_dict, feature_operators, feature_name, operator_dict):
          if any(operator_dict.get(operator, 0) == 1 for operator in feature_operators):
              feature_dict[feature_name] = 1
          else:
              feature_dict[feature_name] = 0
  
  def identify_features(self, sample):
      trig_funs = ['sin', 'cos', 'tan']
      inv_trig_funs = ['arcsin', 'arccos', 'arctan']
      features_single = ['pow2', 'pow3', 'log', 'sqrt', 'exp']

      operator_dict = sample['operator_dict']
      feature_dict = {}

      for feature in features_single:
        feature_dict[feature] = operator_dict[feature]

      self.identify_multiple_features(feature_dict, trig_funs, 'trig', operator_dict)
      self.identify_multiple_features(feature_dict, inv_trig_funs, 'inv_trig', operator_dict)

      sample['feature_dict'] = feature_dict
      return sample

  def generate_random_samples(self, seed=None):
    seed_gen = np.random.RandomState(seed)
    for i in range(self.num_samples):
      sample_seed = seed_gen.randint(1_000_000_000)
      # Number copied somewhere from their github (https://github.com/sdascoli/odeformer/blob/c9193012ad07a97186290b98d8290d1a177f4609/odeformer/trainer.py#L244)
      # TODO: May need to set with more care?
      self.env.rng = np.random.RandomState(sample_seed)
      sample, errors = self.env.gen_expr(train=True)
      sample = self.identify_operators(sample)
      sample = self.identify_features(sample)

      # Set filename
      sample_filename = f"sample_{sample_seed}.pt"
      sample_filepath = os.path.join(self.samples_path, sample_filename)

      # Save file using pickle
      with open(sample_filepath, 'wb') as f:
        pickle.dump(sample, f)
      # print(f"[INFO] Saved to {sample_filepath}")

    print(f"[INFO] Data generation complete. Generated {self.num_samples} samples, saved to {self.samples_path}")


class ManualSamplesGenerator():
  def __init__(self, samples_path='/content/drive/MyDrive/aisc/samples'): # TODO: update with more parameters
    self.samples_path = samples_path
    os.makedirs(self.samples_path, exist_ok=True)

  def generate_exponential_samples(self):
    pass # TODO: copy functionality from Axel's notebook
  
  def generate_hyperbolic_samples(self):
    pass # TODO: copy functionality from Axel's notebook
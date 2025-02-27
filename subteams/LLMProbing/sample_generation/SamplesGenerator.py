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
  def __init__(self, samples_path='/content/drive/MyDrive/aisc/samples', num_samples=10, seed=None): # TODO: update with parameters to specify operators / dimension etc.
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

  def generate_random_samples(self):
    seed_gen = np.random.RandomState(self.seed)
    for i in range(self.num_samples):
      sample_seed = seed_gen.randint(1_000_000_000)
      # Number copied somewhere from their github (https://github.com/sdascoli/odeformer/blob/c9193012ad07a97186290b98d8290d1a177f4609/odeformer/trainer.py#L244)
      # TODO: May need to set with more care?
      self.env.rng = np.random.RandomState(sample_seed)
      sample, errors = self.env.gen_expr(train=True)
      sample = self.identify_operators(sample)
      sample = self.identify_features(sample)

      # Set filename
      sample_filename = f"sample_random_{sample_seed}.pt"
      sample_filepath = os.path.join(self.samples_path, sample_filename)

      # Save file using pickle
      with open(sample_filepath, 'wb') as f:
        pickle.dump(sample, f)
      # print(f"[INFO] Saved to {sample_filepath}")

    print(f"[INFO] Data generation complete. Saved {self.num_samples} samples to {self.samples_path}")


class ManualSamplesGenerator():
  def __init__(self, samples_path='/content/drive/MyDrive/aisc/samples'): # TODO: update with more parameters (TBD)
    self.samples_path = samples_path
    os.makedirs(self.samples_path, exist_ok=True)

  def clean_expression(self, expression):
    # In order to make expressions for equations easier to read
    cleaned = expression.replace('--', '')
    cleaned = cleaned.replace(' -', '-')
    cleaned = cleaned.replace('- ', '-')
    return cleaned

  def save_generated_samples(self, samples, template='sample_man_'):
    for idx, sample in enumerate(samples):
        sample_filename = template + f"{idx}.pt"
        sample_filepath = os.path.join(self.samples_path, sample_filename)
        os.makedirs(os.path.dirname(sample_filepath), exist_ok=True)
        # Save file using pickle
        with open(sample_filepath, 'wb') as f:
          pickle.dump(sample, f)
        # print(f"[INFO] Saved to {sample_filepath}")

  def generate_exponential_samples(self, t_values, c_values, a_values):
    manual_samples = []

    for c_val in c_values:
        for a_val in a_values:
            trajectory = (c_val * np.exp(-a_val * t_values)).reshape(-1, 1)
            sample_dict = {
                'times': t_values,
                'trajectory': trajectory,
                'a': float(a_val),  # Convert to float for better serialization
                'c': float(c_val)   # Convert to float for better serialization
                ,'feature_dict': {"exponential_decay": 1, "hyperbolic": 0}
                ,'expression': self.clean_expression(f"{c_val} * np.exp(-{a_val} * t)")
            }
            manual_samples.append(sample_dict)

    self.save_generated_samples(manual_samples, template='sample_exp_')
    num_samples = len(c_values) * len(a_values)
    print(f'[INFO] Data generation complete. Saved {num_samples} exponential samples to {self.samples_path}')
  
  def generate_hyperbolic_samples(self, t_values, c_values, t0_values):
    manual_samples = []

    for c_val in c_values:
        for t0_val in t0_values:
            trajectory = (c_val / (t0_val - t_values)).reshape(-1, 1)
            sample_dict = {
                'times': t_values,
                'trajectory': trajectory,
                'a': float(t0_val),  # Convert to float for better serialization
                'c': float(c_val)   # Convert to float for better serialization
                ,'feature_dict': {"exponential_decay": 0, "hyperbolic": 1} # TODO: think about how we could make this more general for the future
                ,'expression': self.clean_expression(f"{c_val} / ({t0_val} - t)")
            }
            manual_samples.append(sample_dict)

    self.save_generated_samples(manual_samples, template='sample_hyp_')
    num_samples = len(c_values) * len(t0_values)
    print(f'[INFO] Data generation complete. Saved {num_samples} hyperbolic samples to {self.samples_path}')
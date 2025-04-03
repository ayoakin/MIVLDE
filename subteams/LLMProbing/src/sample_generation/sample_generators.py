import os
import numpy as np
import pickle
from tqdm import tqdm
from tqdm.contrib import itertools
try:
    from odeformer.envs.environment import FunctionEnvironment
    from parsers import get_parser
except ModuleNotFoundError as e:
    print("[ERROR] Could not import odeformer. Check path and installation.")
    raise e

class RandomSamplesGenerator():
  '''
  Generate samples using odeformer's sample generation

  TODO: update this description

  Params:
  * operators, (id, add, sub, mul, div, abds, inv, sqrt, log, exp, sin, arcsin, cos, arccos, tan, arctan, pow2, pow3)
  * their relative frequencies,
  * ode system dimension,
  * number of samples,
  * path for saving samples

  Notes:

  1. operator_probability = operator_frequency/sum(frequencies),
  where the sum is over binary or unary operators. The probabilities appear here in odeformer/envs/generators.py line 411.
  2. Some operators overlap, e.g. mul can create positive powers, as can pow2 and pow3, while div can create negative powers, as can inv. If you wanted to allow only powers of 2, then don't include mul or pow3 in operators_to_use.
  3. Including 'pow' in operators_to_use doesn't seem to do anything, e.g. there are no powers in the samples if operators_to_use = "id:1,add:1,pow:1".

  Each sample is a dictionary with entries:
  1. times,
  2. trajectory,
  3. tree_encoded: prefix notation, as list of operators and exact constants
  4. skeleton_tree_encoded: same as above, but with 'CONSTANT' instead of constants' values
  5. tree
  6. skeleton_tree: same as (4) but normal maths expression rather than prefix notation
  7. infos: number of points, number of unary and binary operators, dimension
  8. operator_dict: dictionary of which operators the sample includes, from the list 'sin', 'cos', 'arcsin', 'arccos', 'log', 'exp', 'tan', 'arctan'.
  9. feature_dict: dictionary of which features the sample includes

  '''
  def __init__(self): 
    pass

  def identify_operators(self, sample):
      all_operators = {'sin', 'cos', 'arcsin', 'arccos', 'log', 'exp', 'tan', 'arctan'}

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
      # Group operators to consider as a single feature
      sin_cos = ['sin', 'cos']
      arc_sin_cos = ['arcsin', 'arccos']
      
      # Each of these operators is considered as a feature
      features_single = ['log', 'exp', 'tan', 'arctan']
      
      # Each of these operators is checked for in the skeleton_tree representation of the sample, because they don't come uniquely from one operator
      features_tree_search = {'pow2' : '**2', 'pow3' : '**3', 'inv' : '**-1', 'sqrt' : '**1 * (2)**-1'} # maybe should move these to operator_dict

      skeleton_tree = str(sample['skeleton_tree'])
      operator_dict = sample['operator_dict']
      feature_dict = {}

      for feature in features_single:
        feature_dict[feature] = operator_dict[feature]

      self.identify_multiple_features(feature_dict, sin_cos, 'sin_cos', operator_dict)
      self.identify_multiple_features(feature_dict, arc_sin_cos, 'arc_sin_cos', operator_dict)

      for key, value in features_tree_search.items():
        feature_dict[key] = int(value in skeleton_tree)

      sample['feature_dict'] = feature_dict
      return sample

  def generate_random_samples(self, samples_path, seed=None, num_samples=10, operators_to_use='id:1,add:1,mul:1', min_dimension=1, max_dimension=1, sample_descriptor='random'):
    os.makedirs(samples_path, exist_ok=True)
    parser = get_parser()
    params = parser.parse_args(args=["--operators_to_use", operators_to_use, "--min_dimension", str(min_dimension), "--max_dimension", str(max_dimension)])
    env = FunctionEnvironment(params)

    seed_gen = np.random.RandomState(seed)
    for i in tqdm(range(num_samples), desc=f'Generating {sample_descriptor} samples'):
      sample_seed = seed_gen.randint(1_000_000_000)

      # Set filename
      sample_filename = f"sample_{sample_descriptor}_{sample_seed}.pt"
      sample_filepath = os.path.join(samples_path, sample_filename)

      if not os.path.exists(sample_filepath):
        # Number copied somewhere from their github (https://github.com/sdascoli/odeformer/blob/c9193012ad07a97186290b98d8290d1a177f4609/odeformer/trainer.py#L244)
        # TODO: May need to set with more care?
        env.rng = np.random.RandomState(sample_seed)
        sample, errors = env.gen_expr(train=True)
        sample = self.identify_operators(sample)
        sample = self.identify_features(sample)

        # Save file using pickle
        with open(sample_filepath, 'wb') as f:
          pickle.dump(sample, f)
      else:
         print(f"\nSkipping sample: {sample_descriptor}_{sample_seed} (sample file already exists)")

    print(f"\n[INFO] Data generation complete. Saved {num_samples} {sample_descriptor} samples to {samples_path}")


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
    for idx, sample in tqdm(enumerate(samples), desc='Saving generated samples'):
        sample_filename = template + f"{idx}.pt"
        sample_filepath = os.path.join(self.samples_path, sample_filename)
        os.makedirs(os.path.dirname(sample_filepath), exist_ok=True)
        # Save file using pickle
        with open(sample_filepath, 'wb') as f:
          pickle.dump(sample, f)

  def generate_exponential_samples(self, t_values, c_values, a_values):
    manual_samples = []

    for c_val, a_val in itertools.product(c_values, a_values, desc='Generating exponential samples'):
      trajectory = (c_val * np.exp(-a_val * t_values)).reshape(-1, 1)
      derivative = (-a_val * c_val * np.exp(-a_val * t_values))
      derivative_0 = derivative[0]
      derivative_3 = derivative[3]
      sample_dict = {
          'times': t_values,
          'trajectory': trajectory,
          'parameters': {'a': float(a_val), 'c': float(c_val)}, # Convert to float for better serialization
          'feature_dict': {"exponential": 1, "hyperbolic": 0, 'derivative_0' : float(derivative_0), 'derivative_3' : float(derivative_3)},
          'expression': self.clean_expression(f"{c_val} * np.exp(-{a_val} * t)")
      }
      manual_samples.append(sample_dict)

    self.save_generated_samples(manual_samples, template='sample_exp_')
    num_samples = len(c_values) * len(a_values)
    print(f'\n[INFO] Data generation complete. Saved {num_samples} exponential samples to {self.samples_path}')
  
  def generate_hyperbolic_samples(self, t_values, c_values, t0_values):
    manual_samples = []

    for c_val, t0_val in itertools.product(c_values, t0_values, desc='Generating hyperbolic samples'):
      trajectory = (c_val / (t0_val - t_values)).reshape(-1, 1)
      derivative = (c_val / (t0_val - t_values)**2)
      derivative_0 = derivative[0]
      derivative_3 = derivative[3]
      sample_dict = {
          'times': t_values,
          'trajectory': trajectory,
          'parameters': {'t0': float(t0_val), 'c': float(c_val)}, # Convert to float for better serialization
          'feature_dict': {"exponential": 0, "hyperbolic": 1, "derivative_0" : float(derivative_0), 'derivative_3' : float(derivative_3)},
          'expression': self.clean_expression(f"{c_val} / ({t0_val} - t)")
      }
      manual_samples.append(sample_dict)

    self.save_generated_samples(manual_samples, template='sample_hyp_')
    num_samples = len(c_values) * len(t0_values)
    print(f'\n[INFO] Data generation complete. Saved {num_samples} hyperbolic samples to {self.samples_path}')
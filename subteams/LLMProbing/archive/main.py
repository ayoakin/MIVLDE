try:
    import odeformer
    from odeformer.model import SymbolicTransformerRegressor
except ImportError:
    print("Error: Please install odeformer package first using 'pip install odeformer'")
    exit(1)

try:
    dstr = SymbolicTransformerRegressor(from_pretrained=True)
    model_args = {'beam_size': 50, 'beam_temperature': 0.1}
    dstr.set_model_args(model_args)
except Exception as e:
    print(f"Error initializing the model: {str(e)}")
    exit(1)

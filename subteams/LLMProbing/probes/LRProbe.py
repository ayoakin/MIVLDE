import torch

class LRProbe(torch.nn.Module):
    def __init__(self, d_in=512): # Default decoder layer activation dimension
        super(LRProbe, self).__init__()
        self.hidden = torch.nn.Linear(d_in, 1, bias=False)

    def forward(self, x):
        x = self.hidden(x)
        x = x.squeeze(-1)
        return x

    def predict(self, acts):
        with torch.no_grad():
            return self(acts)

    def direction(self):
        return self.hidden.weight.data[0]
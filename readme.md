## BPN: Bi-level Physics-Informed Neural Networks for PDE Constrained Optimization using Broyden's Hypergradients



Code for paper "BPN: Bi-level Physics-Informed Neural Networks for PDE Constrained Optimization using Broyden's Hypergradients".

## Requirements

```python
pytorch>=1.11.0
scipy>=1.8.1
termcolor>=1.1.0
deepxde>=1.0.0

```

## Usage
Running PDECO on 1d Poisson's demo problem, 
```python
python poisson_example.py --ft_steps 64 --ift_method broyden --threshold 32 --gpu 0
```
Running PDECO on Poisson's 2d CG  problem,

```python
python poisson_ball_domain.py --ft_steps 256 --ift_method broyden --threshold 32 --gpu 0			
```







## Solutions for possible bugs about DeepXDE

Our code relies on deepxde>1.0.0. However, there are some bugs or imcompatible code in current version of DeepXde. Please modify some source code as follows, otherwise you might encounter bugs or unpredictable results.

If you encounter the following errors:

**A.**

```bash
...
File "ENV_PATH/lib/python3.9/site-packages/torch/utils/data/distributed.py", line 99, in __iter__
    indices = torch.randperm(len(self.dataset), generator=g).tolist()
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
```

Please modify `ENV_PATH/lib/python3.9/site-packages/torch/utils/data/distributed.py` (line 97):

```python
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
```

to:

```python
            # deterministically shuffle based on epoch and seed
            g = torch.Generator(device="cuda") # add this
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
```

**B.**

```bash
...
  File "ENV_PATH/lib/python3.9/site-packages/deepxde/model.py", line 225, in outputs_losses
    outputs_ = self.net(self.net.inputs)
...
  File "ENV_PATH/lib/python3.9/site-packages/torch/nn/functional.py", line 1848, in linear
    return torch._C._nn.linear(input, weight, bias)
RuntimeError: expected scalar type Float but found Double
```

Please modify `ENV_PATH/lib/python3.9/site-packages/deepxde/model.py` (line 225):

```python
            self.net.train(mode=training)
            self.net.inputs = torch.as_tensor(inputs)
            self.net.inputs.requires_grad_()
            outputs_ = self.net(self.net.inputs)
```

to:

```python
            self.net.train(mode=training)
            self.net.inputs = torch.as_tensor(inputs)
            self.net.inputs.requires_grad_()
            outputs_ = self.net(self.net.inputs.float()) # add this
```

**C**.

Modify ```deepxde.data.test_points(boundary=True)```

**D.** 

Add `capturable=True` to `deepxde/optimizers/pytorch/optimizers` .







 
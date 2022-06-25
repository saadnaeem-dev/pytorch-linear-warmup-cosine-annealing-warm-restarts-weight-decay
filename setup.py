from setuptools import setup

setup(
	name="linear_warmup_cosine_annealing_warm_restarts_weight_decay",
	version="1.0",
	author="Saad Naeem",
	packages=['linear_warmup_cosine_annealing_warm_restarts_weight_decay'],
	description= "CosineAnnealingWarmRestarts with initial linear warmup to n steps followed by wight decay in consecutive cycles fo PyTorch",
	long_description=open("README.md").read(),
)
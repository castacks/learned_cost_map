from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['learned_cost_map', 'learned_cost_map.utils', 'learned_cost_map.trainer', 'learned_cost_map.terrain_utils', 'learned_cost_map.dataloader'],
  package_dir={'': 'scripts'}
)

setup(**d)
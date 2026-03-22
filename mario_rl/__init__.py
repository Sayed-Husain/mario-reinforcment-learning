"""Mario RL: reinforcement learning agents for Super Mario Bros."""

# Suppress the gym deprecation warning. We use gymnasium for our own code,
# but gym-super-mario-bros still imports gym internally.
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

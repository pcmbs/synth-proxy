"""
Template for the internal representation additional synthesizers, 
implemented as a tuple of SynthParameter instances.

Remarks:
- Each parameter should be represented as a SynthParameter instance
(see `./src/utils/synth/synth_parameter.py` for available arguments).
- The process of manually adding and restricting the parameters is left to the user
but can be made easier using the `./src/utils/synth/get_synth_parameters.py` script.
"""

from utils.synth import SynthParameter

SYNTH_NAME = "new-wonderful-synth"

SYNTH_PARAMETERS = (
    # SynthParameter(index=0, ...),
    # SynthParameter(index=1, ...),
    # SynthParameter(index=2, ...),
    # SynthParameter(index=3, ...),
    # SynthParameter(index=4, ...),
    ...
)

if __name__ == "__main__":
    pass

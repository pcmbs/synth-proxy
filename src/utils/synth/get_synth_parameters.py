# pylint: disable=W0212:protected-access
"""
Script to print all parameters of a synthesizer as they should appear in
`./src/data/synths/my_synth.py`

Note that this script is only meant to ease the process of integrating a new synthesizer, not replacing it:
- All arguments of all SynthParameter instances should be double-checked.  
- Other SynthParameter arguments (like `default_value`, `interval`, or `excluded_cat_idx`) 
should be added as needed.
"""
import os

from dotenv import load_dotenv
from pedalboard import load_plugin
from pedalboard._pedalboard import to_python_parameter_name

load_dotenv()

# replace with corresponding environment variable or plugin path
VST3_PLUGIN_PATH = os.environ["DIVA_PATH"]


if __name__ == "__main__":
    synth = load_plugin(
        path_to_plugin_file=VST3_PLUGIN_PATH,
        plugin_name="my_plugin",
    )
    print("\n Synth Parameters: \n")

    all_params = []

    for i, p in enumerate(synth._parameters):
        if not (p.name.startswith("CC ") or p.name.startswith("Program")):
            all_params.append(p)

            try:
                is_float = isinstance(float(p.string_value), float)
            except ValueError:
                is_float = False

            cardinality = p.num_steps

            if cardinality == 2:
                param_type = "bin"
            else:
                param_type = "num" if is_float else "cat"

            cardinality_arg = f"{cardinality}" if cardinality <= 100 else -1

            name = to_python_parameter_name(p)

            print(
                f'SynthParameter(index={i}, name="{name}", type_="{param_type}", '
                f"default_value={p.raw_value}, cardinality={cardinality_arg}),"
            )

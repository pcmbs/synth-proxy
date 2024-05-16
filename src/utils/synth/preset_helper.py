"""
Helper class uses to facilitate: 
- the synthetic presets generation process by grouping synthesizer parameters by types, cardinality, etc., 
  such that values for parameters with the same characteristics can be sampled simultaneously. 
- the preset input encoding, by making the parameter types, cardinalities, etc., easily accessible 
  to the class implementing the input encoding function.

Synthesizer are internally represented as a sequence of SynthParameter instances. 
The sequence specific to a synthesizer can be found in `./src/data/synths/<synth-name>.py
"""

from typing import List, Sequence, Tuple
import data.synths


class PresetHelper:
    """
    Helper class uses to facilitate:

    - the synthetic presets generation process by grouping synthesizer parameters by types, cardinality, etc.,
    such that values for parameters with the same characteristics can be sampled simultaneously.
    - the preset input encoding, by making the parameter types, cardinalities, etc., easily accessible
    to the class implementing the input encoding function.

    Synthesizer are internally represented as a sequence of SynthParameter instances.
    The sequence specific to a synthesizer can be found in `./src/data/synths/<synth-name>.py
    """

    def __init__(
        self,
        synth_name: str = "talnm",
        parameters_to_exclude: Sequence[str] = (),
    ):
        """
        Helper class for a given synthesizer used to facilitate the synthetic data generation process
        and presets input encodings.

        Args:
            synth_name (str): name of the synthesizer. This will allow to retrieve the corresponding synthesizer representation
            from `./src/data/synths`.
            parameters_to_exclude (Sequence[str]): list of parameters to exclude, i.e, parameters that are kept to their default value
            during the data generation process and are not inputs to the preset encoder. Can be the full name or a pattern
            that appears at the {begining, end}. In the latter, the pattern must be {followed, preceded} by a "*". (default: ())
        """
        if synth_name in ["talnm", "dexed", "diva"]:
            parameters = getattr(data.synths, synth_name).SYNTH_PARAMETERS
        else:
            raise NotImplementedError()

        self._synth_name = synth_name
        self._parameters = parameters
        self._excl_parameters_str = parameters_to_exclude

        self._excl_parameters = self._exclude_parameters(parameters_to_exclude)

        self._excl_parameters_idx = [p.index for p in self._excl_parameters]
        self._excl_parameters_val = [p.default_value for p in self._excl_parameters]

        self._used_parameters = [p for p in self._parameters if p.index not in self._excl_parameters_idx]
        self._used_parameters_idx = [p.index for p in self._used_parameters]
        self._used_parameters_descr = [
            (i, p.index, p.name, p.type) for i, p in enumerate(self._used_parameters)
        ]
        self._used_num_parameters_idx = [p[0] for p in self._used_parameters_descr if p[3] == "num"]
        self._used_bin_parameters_idx = [p[0] for p in self._used_parameters_descr if p[3] == "bin"]
        self._used_cat_parameters_idx = [p[0] for p in self._used_parameters_descr if p[3] == "cat"]
        self._used_noncat_parameters_idx = sorted(
            self._used_num_parameters_idx + self._used_bin_parameters_idx
        )

        self._relative_idx_from_name = {p.name: i for i, p in enumerate(self._used_parameters)}

        assert len(self._used_num_parameters_idx) + len(self._used_cat_parameters_idx) + len(
            self._used_bin_parameters_idx
        ) == len(self._used_parameters)

        self._grouped_used_parameters = self._group_parameters_for_sampling(self._used_parameters)

    @property
    def synth_name(self) -> str:
        """Return the name of the synthesizer."""
        return self._synth_name

    @property
    def parameters(self) -> List:
        """Return a list of SynthParameter objects of all synthesizer parameters (both used and excluded)."""
        return self._parameters

    @property
    def excl_parameters_idx(self) -> List[int]:
        """Return the absolute indices of the excluded synthesizer parameters."""
        return self._excl_parameters_idx

    @property
    def excl_parameters_val(self) -> List[int]:
        """Return the excluded synthesizer parameters' default value."""
        return self._excl_parameters_val

    @property
    def excl_parameters_str(self) -> Tuple[str]:
        """Return the string patterns used for excluding synthesizer parameters as tuple."""
        return self._excl_parameters_str

    @property
    def used_parameters(self) -> List:
        """
        Return a list of SynthParameter objects of the used synthesizer parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_parameters

    @property
    def num_used_parameters(self) -> int:
        """
        Return the number of used synthesizer parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return len(self._used_parameters_idx)

    @property
    def used_parameters_absolute_idx(self) -> List[int]:
        """
        Return the absolute indices of the used synthesizer parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_parameters_idx

    @property
    def used_noncat_parameters_idx(self) -> List[int]:
        """
        Return the indices of the non-categorical (i.e., numerical and binary) synthesizer parameters
        relative to the used parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_noncat_parameters_idx

    @property
    def used_cat_parameters_idx(self) -> List[int]:
        """
        Return the indices of the categorical synthesizer parameters
        relative to the used parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_cat_parameters_idx

    @property
    def grouped_used_parameters(self) -> dict:
        """
        Return a dictionary of the used synthesizer parameters grouped by types (`continuous` or `discrete`).

        Used synthesizer parameters refer to parameters that are allowed to vary across training samples
        and are thus inputs to the preset encoder.

        - The `continuous` sub-dictionary contains intervals (tuple of floats) as keys and lists of indices
        (relative to the used parameters) of numerical synthesizer parameterssharing that interval as values.
        Note: Categorical and binary synthesizer parameters are inherently discrete and are not included
        in the `continuous` sub-dictionary.

        - The `discrete` sub-dictionary itself contains three sub-dictionaries: 'num', 'cat', and 'bin'.
        Each of these sub-dictionaries has tuples (cat_values, cat_weights) as keys,
        where `cat_values` and `cat_weights` are tuples containing possible discrete values
        and associated sampling weights. Similar to the `continuous` sub-dictionary, the values are
        lists of indices (relative to the used parameters) representing discrete synthesizer parameters
        with the same possible values and weights.
        """
        return self._grouped_used_parameters

    @property
    def used_parameters_description(self) -> List[Tuple[int, str]]:
        """Return the description of the used synthesizer parameters as a
        list of tuple (<idx>, <synth-param-idx>, <synth-param-name>, <synth-param-type>)."""
        return self._used_parameters_descr

    def relative_idx_from_name(self, name: str) -> int:
        """
        Return the index of the synthesizer parameter relative to the used parameters given its name
        or None if not found (meaning it is either excluded or wrong)."""
        return self._relative_idx_from_name.get(name, None)

    def _exclude_parameters(self, pattern_to_exclude: Sequence[str]) -> list[int]:
        def match_pattern(name, pattern):
            return (
                (name == pattern)
                or (pattern.endswith("*") and name.startswith(pattern[:-1]))
                or (pattern.startswith("*") and name.endswith(pattern[1:]))
                or (pattern.startswith("*") and pattern.endswith("*") and name.find(pattern[1:-1]) != -1)
            )

        return [
            p
            for p in self._parameters
            if any(match_pattern(p.name, pattern) for pattern in pattern_to_exclude)
        ]

    def _group_parameters_for_sampling(self, parameters):
        grouped_parameters = {"continuous": {}, "discrete": {"num": {}, "cat": {}, "bin": {}}}

        for i, p in enumerate(parameters):
            type_key = p.type
            if p.cardinality == -1:
                key = p.interval
                if key in grouped_parameters["continuous"]:
                    grouped_parameters["continuous"][key].append(i)
                else:
                    grouped_parameters["continuous"][key] = [i]
            else:
                vw_key = (p.cat_values, p.cat_weights)
                if vw_key in grouped_parameters["discrete"][type_key]:
                    grouped_parameters["discrete"][type_key][vw_key].append(i)
                else:
                    grouped_parameters["discrete"][type_key][vw_key] = [i]

        return grouped_parameters


if __name__ == "__main__":
    SYNTH = "talnm"

    if SYNTH == "talnm":
        PARAMETERS_TO_EXCLUDE_STR = (
            "master_volume",
            "voices",
            "lfo_1_sync",
            "lfo_1_keytrigger",
            "lfo_2_sync",
            "lfo_2_keytrigger",
            "envelope*",
            "portamento*",
            "pitchwheel*",
            "delay*",
        )

    elif SYNTH == "diva":
        PARAMETERS_TO_EXCLUDE_STR = (
            "main:output",
            "vcc:*",
            "opt:*",
            "scope1:*",
            "clk:*",
            "arp:*",
            "plate1:*",
            "delay1:*",
            "chrs2:*",
            "phase2:*",
            "rtary2:*",
            "*keyfollow",
            "*velocity",
            "env1:model",
            "env2:model",
            "*trigger",
            "*release_on",
            "env1:quantise",
            "env2:quantise",
            "env1:curve",
            "env2:curve",
            "lfo1:sync",
            "lfo2:sync",
            "lfo1:restart",
            "lfo2:restart",
            "mod:rectifysource",
            "mod:invertsource",
            "mod:addsource*",
            "*revision",
            "vca:pan",
            "vca:volume",
            "vca:vca",
            "vca:panmodulation",
            "vca:panmoddepth",
            "vca:mode",
            "vca:offset",
        )

    elif SYNTH == "dexed":
        PARAMETERS_TO_EXCLUDE_STR = (
            "cutoff",
            "resonance",
            "output",
            "master_tune_adj",
            "*_key_sync",
            "middle_c",
            "*_switch",
            "*_break_point",
            "*_scale_depth",
            "*_key_scale",
            "*_rate_scaling",
            "*_key_velocity",
        )

    p_helper = PresetHelper(SYNTH, PARAMETERS_TO_EXCLUDE_STR)

    rnd_sampling_info = p_helper.grouped_used_parameters
    print(f"Total number of used parameters: {p_helper.num_used_parameters}\n")
    print(f"Number of used numerical parameters: {len(p_helper._used_num_parameters_idx)}")
    print(f"Number of used binary parameters: {len(p_helper._used_bin_parameters_idx)}")
    print(f"Number of used categorical parameters: {len(p_helper._used_cat_parameters_idx)}")

    print("Used parameters:")
    for i, param in enumerate(p_helper.used_parameters):
        print(f"{str(i) + ':':<4} {param}")

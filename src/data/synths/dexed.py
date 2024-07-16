"""
Internal representation for the Dexed* synthesizer, 
implemented as a tuple of SynthParameter instances.

*https://github.com/asb2m10/dexed
"""

from utils.synth import SynthParameter

# Only works on windows for now...

SYNTH_NAME = "dexed"

_GENERAL_PARAMETERS = (
    SynthParameter(index=0, name="cutoff", type_="num", default_value=1.0),
    SynthParameter(index=1, name="resonance", type_="num"),
    SynthParameter(index=2, name="output", type_="num", default_value=1.0),
    SynthParameter(index=3, name="master_tune_adj", type_="num", default_value=0.5),
    SynthParameter(index=4, name="algorithm", type_="cat", cardinality=32),
    SynthParameter(index=5, name="feedback", type_="num", cardinality=8),
    SynthParameter(index=6, name="osc_key_sync", type_="bin", default_value=1.0),
    SynthParameter(index=7, name="lfo_speed", type_="num", default_value=0.35),
    SynthParameter(index=8, name="lfo_delay", type_="num", interval=(0.0, 0.5)),
    SynthParameter(index=9, name="lfo_pm_depth", type_="num"),
    SynthParameter(index=10, name="lfo_am_depth", type_="num"),
    SynthParameter(index=11, name="lfo_key_sync", type_="bin", default_value=1.0),
    SynthParameter(
        index=12, name="lfo_wave", type_="cat", cardinality=6, excluded_cat_idx=(5,)
    ),  # exclude S&H for reproducibility
    SynthParameter(
        index=13,
        name="middle_c",
        type_="num",
        cardinality=48,
        default_value=0.5,
    ),
    SynthParameter(
        index=14,
        name="p_mode_sens.",
        type_="num",
        cardinality=8,
        default_value=1.0,
        cat_weights=(21, 1, 1, 1, 1, 1, 1, 1),  # no LFO 75% of the time
    ),  # original default_value: 0.4285714328289032 (exclude?: mod via PMD only ?)
    SynthParameter(index=15, name="pitch_eg_rate_1", type_="num", default_value=1.0),
    SynthParameter(index=16, name="pitch_eg_rate_2", type_="num", default_value=1.0),
    SynthParameter(index=17, name="pitch_eg_rate_3", type_="num", default_value=1.0),
    SynthParameter(index=18, name="pitch_eg_rate_4", type_="num", default_value=1.0),
    SynthParameter(index=19, name="pitch_eg_level_1", type_="num", default_value=0.50),
    SynthParameter(index=20, name="pitch_eg_level_2", type_="num", default_value=0.50),
    SynthParameter(index=21, name="pitch_eg_level_3", type_="num", default_value=0.50),
    SynthParameter(index=22, name="pitch_eg_level_4", type_="num", default_value=0.50),
)

_OPS_PARAMETERS = []
for i in range(6):
    _OPS_PARAMETERS += [
        SynthParameter(index=23 + i * 22, name=f"op{i+1}_eg_rate_1", type_="num", default_value=1.0),
        SynthParameter(index=24 + i * 22, name=f"op{i+1}_eg_rate_2", type_="num", default_value=1.0),
        SynthParameter(index=25 + i * 22, name=f"op{i+1}_eg_rate_3", type_="num", default_value=1.0),
        SynthParameter(
            index=26 + i * 22, name=f"op{i+1}_eg_rate_4", type_="num", default_value=1.0, interval=(0.45, 1.0)
        ),
        SynthParameter(index=27 + i * 22, name=f"op{i+1}_eg_level_1", type_="num", default_value=1.0),
        SynthParameter(index=28 + i * 22, name=f"op{i+1}_eg_level_2", type_="num", default_value=1.0),
        SynthParameter(index=29 + i * 22, name=f"op{i+1}_eg_level_3", type_="num", default_value=1.0),
        SynthParameter(index=30 + i * 22, name=f"op{i+1}_eg_level_4", type_="num"),
        SynthParameter(
            index=31 + i * 22, name=f"op{i+1}_output_level", type_="num", default_value=1.0 if i == 0 else 0.0
        ),
        SynthParameter(  # mod cat_weights to avoid too much
            index=32 + i * 22, name=f"op{i+1}_mode", type_="cat", cardinality=2, cat_weights=(0.75, 0.25)
        ),
        SynthParameter(
            index=33 + i * 22,
            name=f"op{i+1}_f_coarse",
            type_="num",
            cardinality=32,
            default_value=0.032258063554763794,
            cat_weights=(  # avoid to have too much high freqs random samples
                2,
                4,
                1,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.3,
                0.3,
                0.3,
                0.2,
                0.2,
                0.2,
                0.1,
                0.1,
                0.1,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ),
        ),
        SynthParameter(index=34 + i * 22, name=f"op{i+1}_f_fine", type_="num"),
        SynthParameter(
            index=35 + i * 22,
            name=f"op{i+1}_osc_detune",
            type_="num",
            cardinality=15,
            default_value=0.5,
        ),
        SynthParameter(index=36 + i * 22, name=f"op{i+1}_break_point", type_="num", default_value=0.40),
        SynthParameter(index=37 + i * 22, name=f"op{i+1}_l_scale_depth", type_="num"),
        SynthParameter(index=38 + i * 22, name=f"op{i+1}_r_scale_depth", type_="num"),
        SynthParameter(index=39 + i * 22, name=f"op{i+1}_l_key_scale", type_="cat", cardinality=4),
        SynthParameter(index=40 + i * 22, name=f"op{i+1}_r_key_scale", type_="cat", cardinality=4),
        SynthParameter(index=41 + i * 22, name=f"op{i+1}_rate_scaling", type_="num", cardinality=8),
        SynthParameter(index=42 + i * 22, name=f"op{i+1}_a_mod_sens.", type_="num", cardinality=4),
        SynthParameter(index=43 + i * 22, name=f"op{i+1}_key_velocity", type_="num", cardinality=8),
        SynthParameter(index=44 + i * 22, name=f"op{i+1}_switch", type_="bin", default_value=1.0),
    ]


SYNTH_PARAMETERS = _GENERAL_PARAMETERS + tuple(_OPS_PARAMETERS)

del _GENERAL_PARAMETERS
del _OPS_PARAMETERS

if __name__ == "__main__":
    print("")

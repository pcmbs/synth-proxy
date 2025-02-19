"""
Internal representation for the TAL-NoiseMaker* synthesizer, 
implemented as a tuple of SynthParameter instances.

*https://tal-software.com/products/tal-noisemaker
"""

from utils.synth import SynthParameter

SYNTH_NAME = "talnm"

SYNTH_PARAMETERS = (
    # SynthParameter(index=0, name="-", type_="num", default_value=0.5), # not a synthesis parameter
    SynthParameter(index=1, name="master_volume", type_="num", default_value=0.40800002217292786),
    SynthParameter(index=2, name="filter_type", type_="cat", cardinality=12, default_value=0.0),
    SynthParameter(index=3, name="filter_cutoff", type_="num", default_value=1.0),
    SynthParameter(index=4, name="filter_resonance", type_="num"),
    SynthParameter(index=5, name="filter_keyfollow", type_="num"),
    SynthParameter(index=6, name="filter_contour", type_="num"),
    SynthParameter(index=7, name="filter_attack", type_="num"),
    SynthParameter(index=8, name="filter_decay", type_="num"),
    SynthParameter(index=9, name="filter_sustain", type_="num", default_value=1.0),
    SynthParameter(index=10, name="filter_release", type_="num"),
    SynthParameter(index=11, name="amp_attack", type_="num"),
    SynthParameter(index=12, name="amp_decay", type_="num"),
    SynthParameter(index=13, name="amp_sustain", type_="num", default_value=1.0),
    SynthParameter(index=14, name="amp_release", type_="num"),
    SynthParameter(index=15, name="osc_1_volume", type_="num", default_value=0.800000011920929),
    SynthParameter(index=16, name="osc_2_volume", type_="num"),
    SynthParameter(index=17, name="osc_3_volume", type_="num", default_value=0.800000011920929),
    SynthParameter(index=18, name="osc_mastertune", type_="num", default_value=0.5),
    SynthParameter(index=19, name="osc_1_tune", type_="num", default_value=0.25),
    SynthParameter(index=20, name="osc_2_tune", type_="num", default_value=0.5),
    SynthParameter(index=21, name="osc_1_fine_tune", type_="num", default_value=0.5),
    SynthParameter(index=22, name="osc_2_fine_tune", type_="num", default_value=0.5),
    SynthParameter(
        index=23,
        name="osc_1_waveform",
        type_="cat",
        cardinality=3,
        cat_weights=(0.45, 0.45, 0.1),  # not too much noise
    ),
    SynthParameter(index=24, name="osc_2_waveform", type_="cat", cardinality=5),
    SynthParameter(index=25, name="osc_sync", type_="bin"),
    SynthParameter(
        index=26, name="lfo_1_waveform", type_="cat", cardinality=6, excluded_cat_idx=(4,)
    ),  # no S&H waveform for reproducibility
    SynthParameter(
        index=27, name="lfo_2_waveform", type_="cat", cardinality=6, excluded_cat_idx=(4,)
    ),  # no S&H waveform for reproducibility
    SynthParameter(index=28, name="lfo_1_rate", type_="num"),
    SynthParameter(index=29, name="lfo_2_rate", type_="num"),
    SynthParameter(index=30, name="lfo_1_amount", type_="num", default_value=0.5),
    SynthParameter(index=31, name="lfo_2_amount", type_="num", default_value=0.5),
    SynthParameter(index=32, name="lfo_1_destination", type_="cat", cardinality=8),
    SynthParameter(index=33, name="lfo_2_destination", type_="cat", cardinality=8),
    SynthParameter(index=34, name="lfo_1_phase", type_="num"),
    SynthParameter(index=35, name="lfo_2_phase", type_="num"),
    SynthParameter(index=36, name="osc_2_fm", type_="num"),
    SynthParameter(index=37, name="osc_2_phase", type_="num"),
    SynthParameter(index=38, name="osc_1_pw", type_="num", default_value=0.5),
    SynthParameter(index=39, name="osc_1_phase", type_="num", default_value=0.5),
    SynthParameter(
        index=40,
        name="transpose",
        type_="num",
        default_value=0.5,
        cardinality=4,
        cat_values=(0.0, 0.5, 0.75, 1.0),
    ),
    SynthParameter(index=41, name="free_ad_attack", type_="num"),
    SynthParameter(index=42, name="free_ad_decay", type_="num"),
    SynthParameter(index=43, name="free_ad_amount", type_="num"),
    SynthParameter(index=44, name="free_ad_destination", type_="cat", cardinality=6),
    SynthParameter(index=45, name="lfo_1_sync", type_="bin"),
    SynthParameter(index=46, name="lfo_1_keytrigger", type_="bin", default_value=1.0),
    SynthParameter(index=47, name="lfo_2_sync", type_="bin"),
    SynthParameter(index=48, name="lfo_2_keytrigger", type_="bin", default_value=1.0),
    SynthParameter(index=49, name="portamento_amount", type_="num"),
    SynthParameter(index=50, name="portamento_mode", type_="cat", cardinality=3),
    SynthParameter(index=51, name="voices", type_="cat", cardinality=6),
    SynthParameter(index=52, name="velocity_volume", type_="num"),
    SynthParameter(index=53, name="velocity_contour", type_="num"),
    SynthParameter(index=54, name="velocity_filter", type_="num"),
    SynthParameter(index=55, name="pitchwheel_cutoff", type_="num"),
    SynthParameter(index=56, name="pitchwheel_pitch", type_="num"),
    SynthParameter(index=57, name="ringmodulation", type_="num"),
    SynthParameter(index=58, name="chorus_1_enable", type_="bin"),
    SynthParameter(index=59, name="chorus_2_enable", type_="bin"),
    SynthParameter(index=60, name="reverb_wet", type_="num"),
    SynthParameter(
        index=61, name="reverb_decay", type_="num", default_value=0.5, interval=(0.0, 0.6)
    ),  # avoid too long decay
    SynthParameter(index=62, name="reverb_pre_delay", type_="num"),
    SynthParameter(index=63, name="reverb_high_cut", type_="num"),
    SynthParameter(index=64, name="reverb_low_cut", type_="num", default_value=1.0),
    SynthParameter(index=65, name="osc_bitcrusher", type_="num", default_value=1.0),
    SynthParameter(index=66, name="master_high_pass", type_="num"),
    SynthParameter(index=67, name="master_detune", type_="num"),
    SynthParameter(index=68, name="vintage_noise", type_="num"),
    # SynthParameter(index=69, name="panic", type_="bin"), # not a synthesis parameters
    # SynthParameter(index=70, name="midi_learn", type_="bin"), # not a synthesis parameters
    SynthParameter(index=71, name="envelope_destination", type_="cat", cardinality=8),
    SynthParameter(index=72, name="envelope_speed", type_="num", cardinality=6),
    SynthParameter(index=73, name="envelope_amount", type_="num"),
    SynthParameter(index=74, name="envelope_one_shot_mode", type_="bin"),
    SynthParameter(index=75, name="envelope_fix_tempo", type_="bin"),
    SynthParameter(index=76, name="envelope_reset", type_="bin"),
    SynthParameter(index=77, name="filter_drive", type_="num"),
    SynthParameter(index=78, name="delay_wet", type_="num"),
    SynthParameter(index=79, name="delay_time", type_="num", default_value=0.5),
    SynthParameter(index=80, name="delay_sync", type_="bin"),
    SynthParameter(index=81, name="delay_x2_l", type_="bin"),
    SynthParameter(index=82, name="delay_x2_r", type_="bin"),
    SynthParameter(index=83, name="delay_high_shelf", type_="num"),
    SynthParameter(index=84, name="delay_low_shelf", type_="num"),
    SynthParameter(index=85, name="delay_feedback", type_="num", default_value=0.5),
)

if __name__ == "__main__":
    print("")

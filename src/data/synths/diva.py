"""
Internal representation for the Diva* synthesizer, 
implemented as a list of SynthParameter instances.

*https://u-he.com/products/diva/
"""

from utils.synth.synth_parameter import SynthParameter

# Modulation sources for LFOs, OSCs, Filters, Amplifier, MOD-VCO, MOD-Filter, MOD-Feedback
# Included: None, Env1, Env2, LFO1, LFO2, Quantise, Lag, Multiply
# The one depending on user control and the ones not impacting the output sound are excluded (also to simplify a bit the task)
_EXCLUDED_CAT_FOR_MOD = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 22)

# Modulation sources for MODs (rectify, invert, quantise, lag, multiply, and add)
# Included: Env1, Env2, LFO1, LFO2
# The one depending on user control and the ones not impacting the output sound are excluded (also to simplify a bit the task)
_EXCLUDED_CAT_FOR_MOD_GROUP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23)

SYNTH_PARAMETERS = [
    ######################################################################################################
    ###### Main parameters
    SynthParameter(index=0, name="main:output", type_="num", default_value=0.33),
    SynthParameter(index=1, name="main:active_fx1", type_="bin"),
    SynthParameter(index=2, name="main:active_fx2", type_="bin"),
    # SettingsParameter(index=3, name="-:led_colour"),
    ######################################################################################################
    ###### Voice Circuit parameters: exclude all
    # - exclude: all
    # - mod init values: voices to 2, mode to mono, pitchbend to 0
    SynthParameter(index=4, name="vcc:voices", type_="num", cardinality=8),
    SynthParameter(index=5, name="vcc:voice_stack", type_="num", cardinality=6),
    SynthParameter(index=6, name="vcc:mode", type_="cat", default_value=0.25, cardinality=5),
    SynthParameter(index=7, name="vcc:glidemode", type_="cat", cardinality=2),
    SynthParameter(index=8, name="vcc:glide", type_="num"),
    SynthParameter(index=9, name="vcc:glide2", type_="num", default_value=0.5),
    SynthParameter(index=10, name="vcc:gliderange", type_="num", default_value=1.0),
    SynthParameter(index=11, name="vcc:pitchbend_up", type_="num", cardinality=27),
    SynthParameter(index=12, name="vcc:pitchbend_down", type_="num", cardinality=27),
    SynthParameter(index=13, name="vcc:tuningmode", type_="bin"),
    SynthParameter(index=14, name="vcc:transpose", type_="num", default_value=0.25, cardinality=49),
    SynthParameter(index=15, name="vcc:finetunecents", type_="num", default_value=0.5),
    SynthParameter(index=16, name="vcc:note_priority", type_="cat", cardinality=3),
    SynthParameter(index=17, name="vcc:multicore", type_="cat", cardinality=2),
    ######################################################################################################
    ###### Global parameters
    # - exclude: all
    # - mod init values: accuracy to 0.0, offlineacc to 0.0 (high quality not needed)
    # - tuneslop corresponds to detuneAmt
    SynthParameter(index=18, name="opt:accuracy", type_="cat", cardinality=4),
    SynthParameter(index=19, name="opt:offlineacc", type_="cat", cardinality=2),
    SynthParameter(index=20, name="opt:tuneslop", type_="num", default_value=0.25),
    SynthParameter(index=21, name="opt:cutoffslop", type_="num", default_value=0.16),
    SynthParameter(index=22, name="opt:glideslop", type_="num", default_value=0.24),
    SynthParameter(index=23, name="opt:pwslop", type_="num", default_value=0.33),
    SynthParameter(index=24, name="opt:envrateslop", type_="num", default_value=0.26),
    SynthParameter(index=25, name="opt:v1mod", type_="num", default_value=0.5),
    SynthParameter(index=26, name="opt:v2mod", type_="num", default_value=0.5),
    SynthParameter(index=27, name="opt:v3mod", type_="num", default_value=0.5),
    SynthParameter(index=28, name="opt:v4mod", type_="num", default_value=0.5),
    SynthParameter(index=29, name="opt:v5mod", type_="num", default_value=0.5),
    SynthParameter(index=30, name="opt:v6mod", type_="num", default_value=0.5),
    SynthParameter(index=31, name="opt:v7mod", type_="num", default_value=0.5),
    SynthParameter(index=32, name="opt:v8mod", type_="num", default_value=0.5),
    ######################################################################################################
    ###### Envelopes parameters
    # - exclude: velocity, keyfollow, trigger (set to 0.0 (Gate), release_on
    #   maybe model (set to analog or digital?)
    # - attack, decay, sustain, release, velocity, keyfollow are common to all models
    #   while quantise and curve are specific to digital (exclude them?), and release_on is specific to ADS
    #
    ### ENV1
    SynthParameter(index=33, name="env1:attack", type_="num", default_value=0.01),
    SynthParameter(index=34, name="env1:decay", type_="num", default_value=0.5),
    SynthParameter(index=35, name="env1:sustain", type_="num", default_value=1.0),
    SynthParameter(index=36, name="env1:release", type_="num", default_value=0.2, interval=(0.0, 0.5)),
    SynthParameter(index=37, name="env1:velocity", type_="num"),
    SynthParameter(index=38, name="env1:model", type_="cat", cardinality=3, default_value=0.5),
    SynthParameter(index=39, name="env1:trigger", type_="cat", cardinality=4),
    SynthParameter(index=40, name="env1:quantise", type_="bin"),
    SynthParameter(index=41, name="env1:curve", type_="bin"),
    SynthParameter(index=42, name="env1:release_on", type_="bin"),
    SynthParameter(index=43, name="env1:keyfollow", type_="num"),
    ### ENV2
    SynthParameter(index=44, name="env2:attack", type_="num", default_value=0.01),
    SynthParameter(index=45, name="env2:decay", type_="num", default_value=0.5),
    SynthParameter(index=46, name="env2:sustain", type_="num", default_value=1.0),
    SynthParameter(
        index=47,
        name="env2:release",
        type_="num",
        default_value=0.2,
        interval=(0.0, 0.5),
        cardinality=-1,
    ),
    SynthParameter(index=48, name="env2:velocity", type_="num", default_value=0.85),
    SynthParameter(index=49, name="env2:model", type_="cat", cardinality=3, default_value=0.5),
    SynthParameter(index=50, name="env2:trigger", type_="cat", cardinality=4),
    SynthParameter(index=51, name="env2:quantise", type_="bin"),
    SynthParameter(index=52, name="env2:curve", type_="bin"),
    SynthParameter(index=53, name="env2:release_on", type_="bin"),
    SynthParameter(index=54, name="env2:keyfollow", type_="num", default_value=0.24),
    ######################################################################################################
    ###### LFO parameters
    # - exclude: sync (1/4), restart (gate),
    # - restrict {freq, depth}mod_src1 to Env1, Env2, LFO1, LFO2, Multiply, Lag, Quantize
    # - exclude radn_hold and rand_glide from waveform for reproducibility
    ### LFO1
    SynthParameter(index=55, name="lfo1:sync", type_="cat", default_value=0.2689, cardinality=27),
    SynthParameter(index=56, name="lfo1:restart", type_="cat", default_value=0.33, cardinality=4),
    SynthParameter(index=57, name="lfo1:waveform", type_="cat", cardinality=8, excluded_cat_idx=(6, 7)),
    SynthParameter(index=58, name="lfo1:phase", type_="num"),
    SynthParameter(index=59, name="lfo1:delay", type_="num", interval=(0.0, 0.5)),
    SynthParameter(
        index=60,
        name="lfo1:depthmod_src1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=61, name="lfo1:depthmod_dpt1", type_="num"),
    SynthParameter(index=62, name="lfo1:rate", type_="num", default_value=0.5),
    SynthParameter(
        index=63,
        name="lfo1:freqmod_src1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=64, name="lfo1:freqmod_dpt", type_="num", default_value=0.5),
    SynthParameter(index=65, name="lfo1:polarity", type_="cat", cardinality=2),
    ### LFO2
    SynthParameter(index=66, name="lfo2:sync", type_="cat", default_value=0.2689, cardinality=27),
    SynthParameter(index=67, name="lfo2:restart", type_="cat", default_value=0.33, cardinality=4),
    SynthParameter(index=68, name="lfo2:waveform", type_="cat", cardinality=8, excluded_cat_idx=(6, 7)),
    SynthParameter(index=69, name="lfo2:phase", type_="num"),
    SynthParameter(index=70, name="lfo2:delay", type_="num", interval=(0.0, 0.5)),
    SynthParameter(
        index=71,
        name="lfo2:depthmod_src1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=72, name="lfo2:depthmod_dpt1", type_="num"),
    SynthParameter(index=73, name="lfo2:rate", type_="num", default_value=0.5),
    SynthParameter(
        index=74,
        name="lfo2:freqmod_src1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=75, name="lfo2:freqmod_dpt", type_="num", default_value=0.5),
    SynthParameter(index=76, name="lfo2:polarity", type_="cat", cardinality=2),
    ######################################################################################################
    ###### MOD parameters
    # - exclude: rectify (none), invert (none), add (source1 & 2 to None)
    # - restrict quantize, lag, and multiply sources to ENV1&2, LFO1&2
    SynthParameter(index=77, name="mod:quantise", type_="num"),
    SynthParameter(index=78, name="mod:slew_rate", type_="num", default_value=0.5),
    SynthParameter(
        index=79,
        name="mod:rectifysource",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=80,
        name="mod:invertsource",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=81,
        name="mod:quantisesource",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=82,
        name="mod:lagsource",
        type_="cat",
        default_value=0.0,
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=83,
        name="mod:addsource1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=84,
        name="mod:addsource2",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=85,
        name="mod:mulsource1",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    SynthParameter(
        index=86,
        name="mod:mulsource2",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD_GROUP,
    ),
    ######################################################################################################
    ###### OSC parameters
    # Avalaible oscillator types: Triple VCO (0), Dual VCO (1), DCO (2), Dual VCO Eco (3), Digital (4)
    SynthParameter(index=87, name="osc:model", type_="cat", cardinality=5),
    # tune1 shared amongst all osc types,
    # tune2 shared amongst osc type 0-1-3-4 (it is considered as a continuous parameters for Digital)
    # tune3 is only active on the Triple VCO
    SynthParameter(index=88, name="osc:tune1", type_="num", default_value=0.5, cardinality=5),
    SynthParameter(index=89, name="osc:tune2", type_="num", default_value=0.5),
    SynthParameter(index=90, name="osc:tune3", type_="num", default_value=0.5, cardinality=5),
    # vibrato is accessible via the Tuning section
    SynthParameter(index=91, name="osc:vibrato", type_="num"),
    # pulsewidth shared amongst osc type 2-3-4-5.
    # Note that for d-osc this parameters has different functions depending on the
    # choose waveform (see manual pp. 25 for more details)
    SynthParameter(index=92, name="osc:pulsewidth", type_="num", default_value=0.5),
    # shape 1, 2, and 3 controls waveform shape of the Triple VCO only
    SynthParameter(index=93, name="osc:shape1", type_="num", default_value=0.5),
    SynthParameter(index=94, name="osc:shape2", type_="num", default_value=0.5),
    SynthParameter(index=95, name="osc:shape3", type_="num", default_value=0.5),
    # fm is available on Triple VCO, Dual VCO (as Cross Mod), and Digital (as Cross)
    SynthParameter(index=96, name="osc:fm", type_="num"),
    # sync is available on Triple VCO, Dual VCO, and Digital
    SynthParameter(index=97, name="osc:sync2", type_="bin"),
    # oscmix is available on Dual VCO, and Digital
    SynthParameter(index=98, name="osc:oscmix", type_="num"),
    # volume1 and volume2 are available on Triple VCO, and Dual VCO Eco
    # volume3 is available on Triple VCO and DCO (as SUB)
    SynthParameter(index=99, name="osc:volume1", type_="num"),
    SynthParameter(index=100, name="osc:volume2", type_="num", default_value=1.0),
    SynthParameter(index=101, name="osc:volume3", type_="num"),
    # pulseshape, sawshape, and suboscshape are available on DCO only.
    # The output being the sum of the three shapes (+ noise)
    SynthParameter(index=102, name="osc:pulseshape", type_="cat", default_value=0.33, cardinality=4),
    SynthParameter(index=103, name="osc:sawshape", type_="cat", cardinality=6),
    SynthParameter(index=104, name="osc:suboscshape", type_="cat", cardinality=6),
    # Tune 1 Mod Source and Depth available on all osc types
    SynthParameter(
        index=105,
        name="osc:tune1modsrc",
        type_="cat",
        default_value=0.652,  # Env2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=106, name="osc:tune1moddepth", type_="num", default_value=0.5),
    # Tune 1 Mod Source and Depth available on Dual VCO, DCO, Dual VCO Eco, and digital
    SynthParameter(
        index=107,
        name="osc:tune2modsrc",
        type_="cat",
        default_value=0.739,  # LFO 2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=108, name="osc:tune2moddepth", type_="num", default_value=0.5),
    # PW Mod Source and Depth available on Dual VCO, DCO, Digital
    SynthParameter(
        index=109,
        name="osc:pwmodsrc",
        type_="cat",
        default_value=0.739,  # LFO 2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=110, name="osc:pwmoddepth", type_="num", default_value=0.5),
    # Shape Mod Source and Depth available on Triple VCO and Digital
    SynthParameter(
        index=111,
        name="osc:shapesrc",
        type_="cat",
        default_value=0.739,  # LFO 2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=112, name="osc:shapedepth", type_="num", default_value=0.5),
    # 113-121 are Waveforms for Dual VCO (waveforms get added if several are on)
    SynthParameter(index=113, name="osc:triangle1on", type_="bin"),
    SynthParameter(index=114, name="osc:sine2on", type_="bin"),
    SynthParameter(index=115, name="osc:saw1on", type_="bin", default_value=1.0),
    # Activate Osc1 Pulse for Dual VCO, Mod switch for Osc1 param 2 for Digital
    SynthParameter(index=116, name="osc:pwm1on", type_="bin"),
    SynthParameter(index=117, name="osc:triangle2on", type_="bin"),
    SynthParameter(index=118, name="osc:saw2on", type_="bin"),
    SynthParameter(index=119, name="osc:pulse2on", type_="bin", default_value=1.0),
    # PWM switch for Dual VCO, Mod switch for Osc2 param 2 for Digital
    SynthParameter(index=120, name="osc:pwm2on", type_="bin", default_value=1.0),  # PWW switch
    # noise on with lower probability
    SynthParameter(index=121, name="osc:noise1on", type_="bin", cat_weights=(0.8, 0.2)),
    # Shape model fixed to 0.5 (analog) since can only be noticeable for triangle
    SynthParameter(index=122, name="osc:shapemodel", type_="cat", default_value=0.5, cardinality=3),
    # Sync 3 for Triple VCO only
    SynthParameter(index=123, name="osc:sync3", type_="bin"),
    # Noise volume available for Triple VCO, DCO,
    SynthParameter(index=124, name="osc:noisevol", type_="num"),
    # Noise color available for Triple VCO only
    SynthParameter(index=125, name="osc:noisecolor", type_="cat", cardinality=2),
    # Tune/Shape Mode available for Triple VCO and Digital (only *osc{1,2} for the later)
    SynthParameter(index=126, name="osc:tunemodosc1", type_="bin"),
    SynthParameter(index=127, name="osc:tunemodosc2", type_="bin"),
    SynthParameter(index=128, name="osc:tunemodosc3", type_="bin"),
    SynthParameter(index=129, name="osc:shapemodosc1", type_="bin"),
    SynthParameter(index=130, name="osc:shapemodosc2", type_="bin"),
    SynthParameter(index=131, name="osc:shapemodosc3", type_="bin"),
    # Tune Mod Mode available for Dual VCO only
    SynthParameter(index=132, name="osc:tunemodmode", type_="cat", default_value=1.0, cardinality=4),
    # Eco Wave {1,2} available for Dual VCO Eco only
    SynthParameter(index=133, name="osc:ecowave1", type_="cat", cardinality=4),
    SynthParameter(index=134, name="osc:ecowave2", type_="cat", cardinality=4),
    # Ring Mod Pulse for Digital only
    SynthParameter(index=135, name="osc:ringmodpulse", type_="bin"),
    # in Trimmer panel, exclude
    SynthParameter(index=136, name="osc:drift", type_="num", default_value=0.20),
    # FM mod sources
    SynthParameter(
        index=137,
        name="osc:fmmodsrc",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    # FM Mod Depth: modulate: FM param of Triple VCO and Cross param of Digital;
    #  mirror: Cross Mod Depth of Dual VCO
    SynthParameter(index=138, name="osc:fmmoddepth", type_="num", default_value=0.5),
    # Noise Volume Mod:
    #  - modulate the level of Noise (or the oscillator also responsible for noise) for Triple VCO and DCO
    #  - modulate the osc 1 level of Dual VCO Eco (since responsible for noise)
    #  - modulate Osc Mix of Dual VCO and Digital (since VCO 1 responsible for noise)
    # FIXME: forgot to exclude mod categories for dataset with mn04_all_b...
    SynthParameter(
        index=139,
        name="osc:noisevolmodsrc",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=140, name="osc:noisevolmoddepth", type_="num", default_value=0.5),
    # Digital shape 1-4: Digital Oscillator specific parameters
    SynthParameter(index=141, name="osc:digitalshape2", type_="num"),
    SynthParameter(index=142, name="osc:digitalshape3", type_="num", default_value=0.5),
    SynthParameter(index=143, name="osc:digitalshape4", type_="num"),
    # Waveform selection for Digital
    SynthParameter(index=144, name="osc:digitaltype1", type_="cat", cardinality=7),
    SynthParameter(index=145, name="osc:digitaltype2", type_="cat", cardinality=7),
    # Digital anti-alias (exclude)
    SynthParameter(index=146, name="osc:digitalantialias", type_="bin", default_value=1.0),
    ######################################################################################################
    ###### HPF parameters
    # Available Model: No HPF (VCF Feedback - see param 168), HPF Post, HPF Pre, HPF Bite
    # -> not active for Triple VCO (which has Mixer section instead)
    SynthParameter(index=147, name="hpf:model", type_="cat", cardinality=4),
    # Frequency available for HPF Pre and HPF Bite
    SynthParameter(index=148, name="hpf:frequency", type_="num"),
    # Resonance available for HPF Bite only
    SynthParameter(index=149, name="hpf:resonance", type_="num"),
    # Revision available for HPF Bite only (exclude and fix to 0.0)
    SynthParameter(index=150, name="hpf:revision", type_="cat", cardinality=2),
    # FIXME: didn't find parameter, which anyway is excluded and fixed to 0.0 since depends on midi note
    SynthParameter(index=151, name="hpf:keyfollow", type_="num"),
    # Freq Mod Src and Depth available for HPF Bite only
    SynthParameter(
        index=152,
        name="hpf:freqmodsrc",
        type_="cat",
        default_value=0.652,
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=153, name="hpf:freqmoddepth", type_="num", default_value=0.5),
    # Post HPF Freq available for HPF Post only
    SynthParameter(index=154, name="hpf:post-hpf_freq", type_="num", cardinality=5),
    ######################################################################################################
    ###### VCF parameters
    # Available models: Ladder, Cascade, Multimode, Bite, Uhbie
    SynthParameter(index=155, name="vcf1:model", type_="cat", cardinality=5),
    # Parameters 156-163 available to all models
    SynthParameter(index=156, name="vcf1:frequency", type_="num", default_value=0.6),
    SynthParameter(index=157, name="vcf1:resonance", type_="num"),
    SynthParameter(
        index=158,
        name="vcf1:freqmodsrc",
        type_="cat",
        default_value=0.652,  # ENV 2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=159, name="vcf1:freqmoddepth", type_="num", default_value=0.5),
    SynthParameter(
        index=160,
        name="vcf1:freqmod2src",
        type_="cat",
        default_value=0.739,  # LFO 2
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=161, name="vcf1:freqmod2depth", type_="num", default_value=0.5),
    SynthParameter(index=162, name="vcf1:keyfollow", type_="num"),  # exclude and set to 0.0
    SynthParameter(index=163, name="vcf1:filterfm", type_="num", default_value=0.5),
    # Ladder Mode available for Ladder and Cascade
    SynthParameter(index=164, name="vcf1:laddermode", type_="cat", cardinality=2),
    # Ladder Color only available for Cascade
    SynthParameter(index=165, name="vcf1:laddercolor", type_="cat", cardinality=2, default_value=1.0),
    # Slnky Revision only available for Bite (exclude and set to Rev 1)
    SynthParameter(index=166, name="vcf1:slnkyrevision", type_="cat", cardinality=2),
    # SVF Mode only available for Multimode
    SynthParameter(index=167, name="vcf1:svfmode", type_="cat", cardinality=4),
    # In HPF pannel
    SynthParameter(index=168, name="vcf1:feedback", type_="num", default_value=0.20),
    # Res, Fm, Feedback Mod params available in Mod pannel
    SynthParameter(
        index=169,
        name="vcf1:resmodsrc",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=170, name="vcf1:resmoddepth", type_="num", default_value=0.5),
    SynthParameter(
        index=171,
        name="vcf1:fmamountmodsrc",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=172, name="vcf1:fmamountmoddepth", type_="num", default_value=0.5),
    SynthParameter(
        index=173,
        name="vcf1:feedbackmodsrc",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=174, name="vcf1:feedbackmoddepth", type_="num", default_value=0.5),
    # Shape related parameters only available for Uhbie
    SynthParameter(index=175, name="vcf1:shapemix", type_="num"),
    SynthParameter(
        index=176,
        name="vcf1:shapemodsrc",
        type_="cat",
        default_value=0.739,
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=177, name="vcf1:shapemoddepth", type_="num", default_value=0.5),
    SynthParameter(index=178, name="vcf1:uhbiebandpass", type_="cat", cardinality=2),
    ######################################################################################################
    ###### VCA parameters
    # - excluded: pan, volume, vca, pan modulation
    SynthParameter(index=179, name="vca1:pan", type_="num", default_value=0.5),
    SynthParameter(index=180, name="vca1:volume", type_="num", default_value=0.5),
    SynthParameter(index=181, name="vca1:vca", type_="cat", cardinality=2, default_value=1.0),
    SynthParameter(
        index=182,
        name="vca1:modulation",
        type_="cat",
        default_value=0.696,
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=183, name="vca1:moddepth", type_="num", default_value=0.5),
    SynthParameter(
        index=184,
        name="vca1:panmodulation",
        type_="cat",
        cardinality=24,
        excluded_cat_idx=_EXCLUDED_CAT_FOR_MOD,
    ),
    SynthParameter(index=185, name="vca1:panmoddepth", type_="num", default_value=0.5099999904632568),
    # FIXME: don't know what the two following parameters are
    SynthParameter(index=186, name="vca1:mode", type_="num", default_value=0.0),
    SynthParameter(index=187, name="vca1:offset", type_="num", default_value=0.4399999976158142),
    ######################################################################################################
    ###### Scope parameters (to exclude)
    SynthParameter(index=188, name="scope1:frequency", type_="num", default_value=0.4399999976158142),
    SynthParameter(index=189, name="scope1:scale", type_="num", default_value=0.6000000238418579),
    ######################################################################################################
    ###### FX1 parameters
    # Restrict module to Chorus, Phase, Rotary
    SynthParameter(
        index=190,
        name="fx1:module",
        type_="cat",
        cardinality=5,
        excluded_cat_idx=(2, 3),
    ),
    ### Chorus 1 parameters
    SynthParameter(index=191, name="chrs1:type", type_="cat", default_value=0.5, cardinality=3),
    SynthParameter(index=192, name="chrs1:rate", type_="num", default_value=0.5),
    SynthParameter(index=193, name="chrs1:depth", type_="num", default_value=0.5),
    SynthParameter(index=194, name="chrs1:wet", type_="num", default_value=1.0),
    ### Phase 1 parameters
    # - exclude: stereo, sync, phase
    SynthParameter(index=195, name="phase1:type", type_="cat", cardinality=2),
    SynthParameter(index=196, name="phase1:rate", type_="num", default_value=0.5),
    SynthParameter(index=197, name="phase1:feedback", type_="num"),
    SynthParameter(index=198, name="phase1:stereo", type_="num", default_value=0.5),
    SynthParameter(index=199, name="phase1:sync", type_="bin"),
    SynthParameter(index=200, name="phase1:phase", type_="num"),
    SynthParameter(index=201, name="phase1:wet", type_="num", default_value=1.0),
    SynthParameter(index=202, name="phase1:depth", type_="num", default_value=1.0),
    SynthParameter(index=203, name="phase1:center", type_="num", default_value=0.5),
    ### Reverb 1 (Plate 1) parameters (excluded module)
    SynthParameter(index=204, name="plate1:predelay", type_="num"),
    SynthParameter(index=205, name="plate1:diffusion", type_="num", default_value=1.0),
    SynthParameter(index=206, name="plate1:damp", type_="num", default_value=0.80),
    SynthParameter(index=207, name="plate1:decay", type_="num", default_value=0.5, interval=(0.0, 0.5)),
    SynthParameter(index=208, name="plate1:size", type_="num", default_value=0.75, interval=(0.0, 0.75)),
    SynthParameter(index=209, name="plate1:dry", type_="num", default_value=0.90),
    SynthParameter(index=210, name="plate1:wet", type_="num", default_value=0.40),
    ### Delay 1 parameters (excluded module)
    # - exclude: Side Volume, Left Delay, Right Delay
    SynthParameter(
        index=211, name="delay1:left_delay", type_="num", default_value=0.07, interval=(0.0, 0.25)
    ),
    SynthParameter(
        index=212,
        name="delay1:center_delay",
        type_="num",
        default_value=0.20,
        interval=(0.0, 0.25),
    ),
    SynthParameter(
        index=213,
        name="delay1:right_delay",
        type_="num",
        default_value=0.20,
        interval=(0.0, 0.25),
    ),
    SynthParameter(index=214, name="delay1:side_vol", type_="num", default_value=0.0),
    SynthParameter(index=215, name="delay1:center_vol", type_="num"),
    SynthParameter(index=216, name="delay1:feedback", type_="num", default_value=0.25, interval=(0.0, 0.40)),
    SynthParameter(index=217, name="delay1:hp", type_="num"),
    SynthParameter(index=218, name="delay1:lp", type_="num", default_value=1.0),
    SynthParameter(index=219, name="delay1:dry", type_="num", default_value=1.0),
    SynthParameter(index=220, name="delay1:wow", type_="num", default_value=0.5),
    ### Rotary 1 parameters
    # - exclude: stereo, out, fast, controller, risetime
    SynthParameter(index=221, name="rtary1:mode", type_="cat", cardinality=3),
    SynthParameter(index=222, name="rtary1:mix", type_="num", default_value=1.0),
    SynthParameter(index=223, name="rtary1:balance", type_="num", default_value=0.5),
    SynthParameter(index=224, name="rtary1:drive", type_="num"),
    SynthParameter(index=225, name="rtary1:stereo", type_="num", default_value=1.0),
    SynthParameter(index=226, name="rtary1:out", type_="num", default_value=0.5),
    SynthParameter(index=227, name="rtary1:slow", type_="num", default_value=0.30),
    SynthParameter(index=228, name="rtary1:fast", type_="num", default_value=0.85),
    SynthParameter(index=229, name="rtary1:risetime", type_="num", default_value=0.5),
    SynthParameter(index=230, name="rtary1:controller", type_="cat", cardinality=4),
    ######################################################################################################
    ###### FX2 parameters
    # Restrict module to Plate, Delay
    SynthParameter(
        index=231,
        name="fx2:module",
        type_="cat",
        cardinality=5,
        excluded_cat_idx=(0, 1, 4),
    ),
    ### Chorus 2 parameters (excluded module)
    SynthParameter(index=232, name="chrs2:type", type_="cat", default_value=0.5, cardinality=3),
    SynthParameter(index=233, name="chrs2:rate", type_="num", default_value=0.5),
    SynthParameter(index=234, name="chrs2:depth", type_="num", default_value=0.5),
    SynthParameter(index=235, name="chrs2:wet", type_="num", default_value=1.0),
    ### Phase 2 parameters (excluded module)
    # - exclude: stereo, sync, phase
    SynthParameter(index=236, name="phase2:type", type_="cat", cardinality=2),
    SynthParameter(index=237, name="phase2:rate", type_="num", default_value=0.5),
    SynthParameter(index=238, name="phase2:feedback", type_="num"),
    SynthParameter(index=239, name="phase2:stereo", type_="num", default_value=0.5),
    SynthParameter(index=240, name="phase2:sync", type_="bin"),
    SynthParameter(index=241, name="phase2:phase", type_="num"),
    SynthParameter(index=242, name="phase2:wet", type_="num", default_value=1.0),
    SynthParameter(index=243, name="phase2:depth", type_="num", default_value=1.0),
    SynthParameter(index=244, name="phase2:center", type_="num", default_value=0.5),
    ### Reverb 2 (Plate 2) parameters
    SynthParameter(index=245, name="plate2:predelay", type_="num"),
    SynthParameter(index=246, name="plate2:diffusion", type_="num", default_value=1.0),
    SynthParameter(index=247, name="plate2:damp", type_="num", default_value=0.80),
    SynthParameter(index=248, name="plate2:decay", type_="num", default_value=0.5, interval=(0.0, 0.5)),
    SynthParameter(index=249, name="plate2:size", type_="num", default_value=0.75, interval=(0.0, 0.75)),
    SynthParameter(index=250, name="plate2:dry", type_="num", default_value=0.90),
    SynthParameter(index=251, name="plate2:wet", type_="num", default_value=0.40),
    ### Delay 2 parameters
    # - exclude: Side Volume, Left Delay, Right Delay
    SynthParameter(
        index=252, name="delay2:left_delay", type_="num", default_value=0.07, interval=(0.0, 0.25)
    ),
    SynthParameter(
        index=253,
        name="delay2:center_delay",
        type_="num",
        default_value=0.20,
        interval=(0.0, 0.25),
    ),
    SynthParameter(
        index=254,
        name="delay2:right_delay",
        type_="num",
        default_value=0.20,
        interval=(0.0, 0.25),
    ),
    SynthParameter(index=255, name="delay2:side_vol", type_="num", default_value=0.0),
    SynthParameter(index=256, name="delay2:center_vol", type_="num"),
    SynthParameter(index=257, name="delay2:feedback", type_="num", default_value=0.25, interval=(0.0, 0.40)),
    SynthParameter(index=258, name="delay2:hp", type_="num"),
    SynthParameter(index=259, name="delay2:lp", type_="num", default_value=1.0),
    SynthParameter(index=260, name="delay2:dry", type_="num", default_value=1.0),
    SynthParameter(index=261, name="delay2:wow", type_="num", default_value=0.5),
    ### Rotary 2 parameters (excluded module)
    # - exclude: stereo, out, fast, controller, risetime
    SynthParameter(index=262, name="rtary2:mode", type_="cat", cardinality=3),
    SynthParameter(index=263, name="rtary2:mix", type_="num", default_value=1.0),
    SynthParameter(index=264, name="rtary2:balance", type_="num", default_value=0.5),
    SynthParameter(index=265, name="rtary2:drive", type_="num"),
    SynthParameter(index=266, name="rtary2:stereo", type_="num", default_value=1.0),
    SynthParameter(index=267, name="rtary2:out", type_="num", default_value=0.5),
    SynthParameter(index=268, name="rtary2:slow", type_="num", default_value=0.30),
    SynthParameter(index=269, name="rtary2:fast", type_="num", default_value=0.85),
    SynthParameter(index=270, name="rtary2:risetime", type_="num", default_value=0.5),
    SynthParameter(index=271, name="rtary2:controller", type_="cat", cardinality=4),
    ######################################################################################################
    ###### ARP parameters (excluded)
    SynthParameter(index=272, name="clk:multiply", type_="num", default_value=0.33),
    SynthParameter(index=273, name="clk:timebase", type_="cat", default_value=0.6667, cardinality=4),
    SynthParameter(index=274, name="clk:swing", type_="num"),
    SynthParameter(
        index=275,
        name="arp:direction",
        type_="cat",
        default_value=0.20,
        cardinality=6,
        excluded_cat_idx=(0, 5),  # exclude play since only one note played, and random for reproducibility
    ),
    SynthParameter(index=276, name="arp:octaves", type_="num", cardinality=4),
    SynthParameter(index=277, name="arp:multiply", type_="cat", cardinality=5),  # FIXME: didn't find param
    SynthParameter(index=278, name="arp:restart", type_="cat", cardinality=13),
    # only want arp on a small subset of presets
    SynthParameter(index=279, name="arp:onoff", type_="bin", cat_weights=(0.95, 0.05)),
    SynthParameter(
        index=280, name="arp:order", type_="cat", cardinality=4, excluded_cat_idx=(2, 3)  # only if poly
    ),
]

del _EXCLUDED_CAT_FOR_MOD
del _EXCLUDED_CAT_FOR_MOD_GROUP

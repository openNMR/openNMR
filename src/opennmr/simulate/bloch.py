"""Bloch simulation of shaped (selective) RF pulses.

Ported from ``leonmr/bloch_sim.py`` (L. Gordon). A no-relaxation
(T1 = T2 = infinity) hard-pulse-propagator simulator: each 1 us step of
the shape is treated as a rectangular sub-pulse, and every isochromat
is rotated about the instantaneous effective field (B1x, B1y, offset)
by 2*pi*|Beff|*dt using Rodrigues' rotation formula. No relaxation, no
gradients, one isochromat per offset -- 'offset' is a chemical-shift /
off-resonance frequency, matching PulseOffset in the X-Pulse pulse
programmes.

:func:`generate_shape` and the two BURP tables are an **exact copy** of
the original, which is itself an exact copy of the pulse-programme
family's own ``generate_shape()`` -- kept numerically identical on
purpose, so what is simulated is what the hardware synthesises.

Changed from the original:

* :func:`simulate_pulse` rotates every offset and power scale at once
  (NumPy over the isochromats) instead of one Python loop per
  isochromat. Same arithmetic, same result to floating-point
  precision; hundreds of times faster, which is what makes it usable
  from the GUI.
* :func:`integration_factor` / :func:`peak_b1_for_flip_angle` are new:
  the peak B1 that gives a chosen flip angle, from the shape's coherent
  integration factor (the same ``mean(b1)/max|b1|`` quantity the
  original's table comments use to validate the BURP shapes).
* Plotting draws onto a supplied figure rather than ``pyplot``, so it
  embeds in the GUI.
* **The time step is chosen, not fixed at 1 us** (explicit user request,
  2026-09-30: adiabatic sweeps of several MHz). A frequency-swept pulse
  is delivered as a phase ramp, and a ramp sampled too coarsely is simply
  wrong: at 0.45 turn of phase per step, the WURST/HS inversion profiles
  came out with Mz errors above 1 (measured against a 20x finer step),
  and the old GUI limit of a 999 kHz sweep at 1 us was 0.4995 turn per
  step. :func:`auto_time_step_us` keeps the ramp at <= 0.05 turn per step
  (error < 0.01 in Mz for WURST, HS and tanh/tan at 2 and 5 MHz), and
  stays at the hardware's 1 us whenever that is fine enough -- every
  sweep up to 100 kHz and every amplitude-modulated shape, so those
  results are unchanged. An explicit ``time_step_us`` is always honoured.
* **Adiabatic (frequency-swept) shapes** are new (explicit user request,
  2026-09-27): hyperbolic secant (HSn), WURST-n and tanh/tan -- see
  :func:`adiabatic_shape`. They are not added to :func:`generate_shape`,
  which stays an exact copy of the pulse-programme function: an
  adiabatic pulse's phase depends on its sweep width and duration, which
  ``generate_shape(shape, n_steps)`` does not take. :func:`simulate_pulse`
  routes them itself. Their power is set by an adiabaticity factor Q
  (:func:`peak_b1_for_adiabaticity`), not a flip angle, which they do
  not have.
"""

import numpy as np

# E-BURP-2 (90 deg excitation) and RE-BURP (180 deg refocusing), each as a
# 1000-point SIGNED b1(tau) table (peak-normalised to 1.0, tau uniformly
# spaced over [0,1)) -- exported DIRECTLY from real Bruker TopSpin shape
# files (JCAMP-DX 'Shape Data', ##$SHAPE_PARAMETERS= Type: EBurp2/ReBurp),
# supplied by L. Gordon 07/08/2026. Used instead of hand-entered Fourier-
# coefficient reconstructions to avoid any transcription-error risk.
# CROSS-VALIDATED: recomputing the coherent (signed) integration factor
# (mean(table)/max(|table|)) on each reproduces that shape file's own
# quoted SHAPE_INTEGFAC to 6 significant figures (EBURP2: 6.102960E-02,
# REBURP: 7.981016E-02) -- see x-pulse-sequences' LWG_Selective-
# PulseAcquire_H.py changelog for the full validation story (this same
# cross-check caught and confirmed the fix to that file's power-estimate
# formula, which used to use the WRONG, non-coherent mean(|amp|)).
_EBURP2_TABLE_B1 = (
    0.054881, 0.053111, 0.051408, 0.049769, 0.048195, 0.046685, 0.045236, 0.043849, 0.042521, 0.041253,
    0.040042, 0.038888, 0.037789, 0.036744, 0.035752, 0.034812, 0.033922, 0.033082, 0.032289, 0.031543,
    0.030842, 0.030184, 0.029569, 0.028995, 0.028461, 0.027965, 0.027505, 0.027081, 0.026690, 0.026332,
    0.026005, 0.025707, 0.025437, 0.025193, 0.024974, 0.024778, 0.024604, 0.024450, 0.024316, 0.024198,
    0.024097, 0.024010, 0.023935, 0.023873, 0.023820, 0.023776, 0.023739, 0.023709, 0.023682, 0.023659,
    0.023638, 0.023618, 0.023597, 0.023574, 0.023549, 0.023519, 0.023484, 0.023443, 0.023395, 0.023338,
    0.023272, 0.023196, 0.023109, 0.023010, 0.022899, 0.022774, 0.022635, 0.022482, 0.022313, 0.022128,
    0.021927, 0.021710, 0.021475, 0.021222, 0.020952, 0.020663, 0.020356, 0.020030, 0.019685, 0.019322,
    0.018939, 0.018537, 0.018117, 0.017677, 0.017219, 0.016741, 0.016245, 0.015731, 0.015199, 0.014648,
    0.014080, 0.013494, 0.012892, 0.012273, 0.011637, 0.010986, 0.010319, 0.009637, 0.008940, 0.008230,
    0.007505, 0.006768, 0.006018, 0.005256, 0.004482, 0.003697, 0.002901, 0.002095, 0.001280, 0.000456,
    -0.000377, -0.001218, -0.002066, -0.002921, -0.003783, -0.004651, -0.005525, -0.006403, -0.007286, -0.008174,
    -0.009065, -0.009960, -0.010858, -0.011758, -0.012661, -0.013565, -0.014471, -0.015379, -0.016287, -0.017197,
    -0.018106, -0.019016, -0.019926, -0.020836, -0.021745, -0.022654, -0.023562, -0.024469, -0.025375, -0.026280,
    -0.027184, -0.028086, -0.028987, -0.029886, -0.030784, -0.031680, -0.032574, -0.033466, -0.034357, -0.035245,
    -0.036132, -0.037017, -0.037900, -0.038781, -0.039660, -0.040537, -0.041412, -0.042285, -0.043155, -0.044024,
    -0.044890, -0.045754, -0.046615, -0.047474, -0.048331, -0.049185, -0.050037, -0.050885, -0.051731, -0.052574,
    -0.053413, -0.054249, -0.055082, -0.055911, -0.056736, -0.057558, -0.058375, -0.059187, -0.059995, -0.060797,
    -0.061595, -0.062387, -0.063173, -0.063953, -0.064727, -0.065493, -0.066253, -0.067005, -0.067750, -0.068486,
    -0.069214, -0.069933, -0.070643, -0.071343, -0.072034, -0.072713, -0.073382, -0.074040, -0.074686, -0.075320,
    -0.075942, -0.076551, -0.077147, -0.077729, -0.078297, -0.078851, -0.079390, -0.079914, -0.080423, -0.080916,
    -0.081392, -0.081852, -0.082295, -0.082721, -0.083130, -0.083521, -0.083894, -0.084248, -0.084584, -0.084901,
    -0.085199, -0.085478, -0.085737, -0.085977, -0.086196, -0.086396, -0.086576, -0.086735, -0.086874, -0.086993,
    -0.087090, -0.087168, -0.087224, -0.087260, -0.087274, -0.087268, -0.087241, -0.087193, -0.087124, -0.087034,
    -0.086923, -0.086792, -0.086639, -0.086466, -0.086271, -0.086056, -0.085820, -0.085564, -0.085286, -0.084988,
    -0.084670, -0.084330, -0.083971, -0.083590, -0.083189, -0.082768, -0.082326, -0.081864, -0.081381, -0.080878,
    -0.080354, -0.079810, -0.079245, -0.078659, -0.078053, -0.077426, -0.076778, -0.076109, -0.075419, -0.074707,
    -0.073974, -0.073220, -0.072444, -0.071646, -0.070826, -0.069983, -0.069118, -0.068230, -0.067319, -0.066385,
    -0.065427, -0.064445, -0.063439, -0.062408, -0.061353, -0.060272, -0.059166, -0.058034, -0.056876, -0.055692,
    -0.054480, -0.053242, -0.051976, -0.050682, -0.049359, -0.048008, -0.046628, -0.045219, -0.043781, -0.042312,
    -0.040813, -0.039284, -0.037724, -0.036132, -0.034510, -0.032856, -0.031170, -0.029452, -0.027702, -0.025920,
    -0.024105, -0.022258, -0.020378, -0.018466, -0.016521, -0.014543, -0.012532, -0.010489, -0.008414, -0.006306,
    -0.004166, -0.001995, 0.000209, 0.002444, 0.004710, 0.007007, 0.009334, 0.011692, 0.014079, 0.016495,
    0.018939, 0.021412, 0.023912, 0.026439, 0.028992, 0.031570, 0.034174, 0.036801, 0.039452, 0.042124,
    0.044819, 0.047534, 0.050269, 0.053023, 0.055795, 0.058584, 0.061389, 0.064209, 0.067043, 0.069889,
    0.072748, 0.075618, 0.078497, 0.081385, 0.084280, 0.087182, 0.090089, 0.093001, 0.095915, 0.098831,
    0.101749, 0.104666, 0.107582, 0.110495, 0.113405, 0.116310, 0.119209, 0.122102, 0.124986, 0.127862,
    0.130728, 0.133584, 0.136427, 0.139258, 0.142074, 0.144877, 0.147663, 0.150434, 0.153186, 0.155921,
    0.158637, 0.161334, 0.164009, 0.166664, 0.169297, 0.171907, 0.174494, 0.177058, 0.179597, 0.182110,
    0.184599, 0.187061, 0.189497, 0.191906, 0.194287, 0.196640, 0.198966, 0.201262, 0.203529, 0.205766,
    0.207974, 0.210151, 0.212298, 0.214414, 0.216499, 0.218552, 0.220574, 0.222563, 0.224521, 0.226445,
    0.228337, 0.230195, 0.232020, 0.233812, 0.235569, 0.237292, 0.238980, 0.240634, 0.242252, 0.243834,
    0.245381, 0.246891, 0.248365, 0.249801, 0.251200, 0.252561, 0.253884, 0.255167, 0.256412, 0.257616,
    0.258780, 0.259904, 0.260985, 0.262025, 0.263021, 0.263975, 0.264884, 0.265748, 0.266567, 0.267340,
    0.268065, 0.268743, 0.269372, 0.269952, 0.270481, 0.270959, 0.271386, 0.271759, 0.272078, 0.272343,
    0.272552, 0.272704, 0.272798, 0.272834, 0.272810, 0.272725, 0.272579, 0.272370, 0.272098, 0.271760,
    0.271358, 0.270888, 0.270351, 0.269745, 0.269070, 0.268324, 0.267507, 0.266617, 0.265654, 0.264616,
    0.263504, 0.262315, 0.261050, 0.259707, 0.258285, 0.256785, 0.255204, 0.253543, 0.251800, 0.249976,
    0.248069, 0.246079, 0.244005, 0.241847, 0.239605, 0.237277, 0.234864, 0.232365, 0.229780, 0.227109,
    0.224351, 0.221506, 0.218574, 0.215555, 0.212449, 0.209255, 0.205974, 0.202606, 0.199150, 0.195608,
    0.191978, 0.188261, 0.184458, 0.180568, 0.176592, 0.172530, 0.168382, 0.164149, 0.159830, 0.155427,
    0.150940, 0.146368, 0.141713, 0.136975, 0.132154, 0.127251, 0.122267, 0.117201, 0.112054, 0.106827,
    0.101520, 0.096134, 0.090670, 0.085127, 0.079506, 0.073808, 0.068034, 0.062183, 0.056256, 0.050255,
    0.044178, 0.038028, 0.031804, 0.025506, 0.019136, 0.012694, 0.006180, -0.000406, -0.007062, -0.013789,
    -0.020586, -0.027452, -0.034387, -0.041390, -0.048462, -0.055602, -0.062808, -0.070081, -0.077421, -0.084826,
    -0.092296, -0.099831, -0.107430, -0.115093, -0.122819, -0.130607, -0.138457, -0.146368, -0.154340, -0.162371,
    -0.170461, -0.178608, -0.186814, -0.195075, -0.203392, -0.211763, -0.220188, -0.228664, -0.237191, -0.245768,
    -0.254393, -0.263065, -0.271782, -0.280544, -0.289347, -0.298190, -0.307072, -0.315991, -0.324944, -0.333930,
    -0.342946, -0.351990, -0.361059, -0.370151, -0.379264, -0.388394, -0.397540, -0.406697, -0.415863, -0.425034,
    -0.434208, -0.443381, -0.452550, -0.461711, -0.470860, -0.479993, -0.489108, -0.498198, -0.507262, -0.516294,
    -0.525289, -0.534245, -0.543156, -0.552017, -0.560825, -0.569574, -0.578259, -0.586876, -0.595421, -0.603887,
    -0.612270, -0.620564, -0.628766, -0.636869, -0.644868, -0.652759, -0.660535, -0.668192, -0.675725, -0.683128,
    -0.690396, -0.697524, -0.704506, -0.711338, -0.718015, -0.724530, -0.730880, -0.737059, -0.743063, -0.748886,
    -0.754523, -0.759971, -0.765225, -0.770279, -0.775130, -0.779774, -0.784206, -0.788422, -0.792419, -0.796192,
    -0.799739, -0.803055, -0.806138, -0.808984, -0.811591, -0.813955, -0.816074, -0.817945, -0.819567, -0.820938,
    -0.822054, -0.822916, -0.823520, -0.823867, -0.823955, -0.823783, -0.823350, -0.822656, -0.821700, -0.820484,
    -0.819006, -0.817267, -0.815267, -0.813008, -0.810490, -0.807715, -0.804683, -0.801396, -0.797856, -0.794065,
    -0.790025, -0.785737, -0.781205, -0.776431, -0.771418, -0.766167, -0.760684, -0.754969, -0.749028, -0.742863,
    -0.736477, -0.729875, -0.723060, -0.716035, -0.708806, -0.701375, -0.693748, -0.685927, -0.677917, -0.669723,
    -0.661348, -0.652798, -0.644076, -0.635187, -0.626135, -0.616924, -0.607560, -0.598045, -0.588386, -0.578585,
    -0.568648, -0.558578, -0.548381, -0.538059, -0.527617, -0.517059, -0.506390, -0.495612, -0.484730, -0.473748,
    -0.462668, -0.451496, -0.440234, -0.428885, -0.417452, -0.405940, -0.394351, -0.382687, -0.370952, -0.359149,
    -0.347279, -0.335346, -0.323351, -0.311298, -0.299187, -0.287022, -0.274804, -0.262534, -0.250215, -0.237848,
    -0.225435, -0.212976, -0.200474, -0.187930, -0.175344, -0.162717, -0.150052, -0.137348, -0.124606, -0.111828,
    -0.099013, -0.086164, -0.073279, -0.060360, -0.047408, -0.034423, -0.021406, -0.008357, 0.004723, 0.017834,
    0.030975, 0.044145, 0.057345, 0.070572, 0.083826, 0.097107, 0.110414, 0.123746, 0.137101, 0.150479,
    0.163879, 0.177299, 0.190738, 0.204195, 0.217667, 0.231154, 0.244654, 0.258164, 0.271682, 0.285207,
    0.298736, 0.312267, 0.325797, 0.339323, 0.352842, 0.366352, 0.379849, 0.393330, 0.406792, 0.420230,
    0.433641, 0.447022, 0.460367, 0.473674, 0.486936, 0.500151, 0.513313, 0.526418, 0.539460, 0.552436,
    0.565338, 0.578163, 0.590905, 0.603558, 0.616118, 0.628577, 0.640931, 0.653174, 0.665299, 0.677301,
    0.689174, 0.700912, 0.712509, 0.723958, 0.735253, 0.746389, 0.757359, 0.768157, 0.778777, 0.789213,
    0.799458, 0.809508, 0.819355, 0.828994, 0.838420, 0.847625, 0.856606, 0.865356, 0.873870, 0.882143,
    0.890169, 0.897944, 0.905462, 0.912720, 0.919712, 0.926435, 0.932883, 0.939054, 0.944943, 0.950547,
    0.955862, 0.960886, 0.965615, 0.970048, 0.974180, 0.978011, 0.981538, 0.984760, 0.987675, 0.990283,
    0.992581, 0.994570, 0.996249, 0.997618, 0.998677, 0.999426, 0.999867, 1.000000, 0.999826, 0.999347,
    0.998564, 0.997480, 0.996096, 0.994416, 0.992441, 0.990175, 0.987621, 0.984783, 0.981663, 0.978267,
    0.974598, 0.970660, 0.966458, 0.961996, 0.957280, 0.952315, 0.947106, 0.941658, 0.935976, 0.930067,
    0.923937, 0.917592, 0.911037, 0.904279, 0.897324, 0.890178, 0.882850, 0.875344, 0.867667, 0.859827,
    0.851830, 0.843682, 0.835392, 0.826966, 0.818410, 0.809731, 0.800937, 0.792034, 0.783030, 0.773930,
    0.764743, 0.755474, 0.746130, 0.736719, 0.727246, 0.717718, 0.708141, 0.698523, 0.688868, 0.679184,
    0.669476, 0.659750, 0.650012, 0.640268, 0.630523, 0.620783, 0.611053, 0.601337, 0.591642, 0.581972,
    0.572331, 0.562725, 0.553158, 0.543633, 0.534155, 0.524729, 0.515357, 0.506044, 0.496793, 0.487607,
    0.478490, 0.469444, 0.460473, 0.451580, 0.442766, 0.434034, 0.425387, 0.416827, 0.408355, 0.399974,
    0.391686, 0.383491, 0.375393, 0.367391, 0.359487, 0.351682, 0.343978, 0.336374, 0.328873, 0.321475,
    0.314179, 0.306988, 0.299900, 0.292917, 0.286039, 0.279265, 0.272597, 0.266033, 0.259574, 0.253220,
    0.246971, 0.240826, 0.234785, 0.228847, 0.223013, 0.217282, 0.211653, 0.206126, 0.200700, 0.195376,
    0.190151, 0.185026, 0.180000, 0.175072, 0.170242, 0.165508, 0.160871, 0.156329, 0.151882, 0.147529,
    0.143269, 0.139101, 0.135025, 0.131040, 0.127145, 0.123340, 0.119623, 0.115993, 0.112451, 0.108995,
    0.105624, 0.102338, 0.099136, 0.096017, 0.092979, 0.090024, 0.087148, 0.084353, 0.081636, 0.078998,
    0.076437, 0.073952, 0.071543, 0.069208, 0.066947, 0.064759, 0.062643, 0.060598, 0.058624, 0.056718,
)

_REBURP_TABLE_B1 = (
    -0.074916, -0.074916, -0.074855, -0.074733, -0.074552, -0.074310, -0.074011, -0.073655, -0.073244, -0.072780,
    -0.072266, -0.071702, -0.071093, -0.070442, -0.069749, -0.069020, -0.068258, -0.067465, -0.066645, -0.065801,
    -0.064937, -0.064057, -0.063164, -0.062260, -0.061351, -0.060438, -0.059525, -0.058615, -0.057711, -0.056815,
    -0.055930, -0.055058, -0.054201, -0.053361, -0.052539, -0.051737, -0.050955, -0.050195, -0.049456, -0.048740,
    -0.048045, -0.047372, -0.046720, -0.046087, -0.045474, -0.044878, -0.044298, -0.043732, -0.043178, -0.042635,
    -0.042099, -0.041568, -0.041040, -0.040512, -0.039982, -0.039446, -0.038902, -0.038346, -0.037777, -0.037191,
    -0.036586, -0.035959, -0.035307, -0.034629, -0.033921, -0.033182, -0.032410, -0.031602, -0.030759, -0.029877,
    -0.028956, -0.027995, -0.026993, -0.025950, -0.024866, -0.023741, -0.022574, -0.021367, -0.020120, -0.018833,
    -0.017509, -0.016149, -0.014754, -0.013325, -0.011865, -0.010376, -0.008859, -0.007318, -0.005755, -0.004171,
    -0.002570, -0.000954, 0.000673, 0.002311, 0.003955, 0.005604, 0.007254, 0.008903, 0.010549, 0.012190,
    0.013822, 0.015445, 0.017056, 0.018652, 0.020234, 0.021798, 0.023343, 0.024869, 0.026374, 0.027858,
    0.029319, 0.030757, 0.032171, 0.033562, 0.034930, 0.036274, 0.037595, 0.038894, 0.040170, 0.041424,
    0.042658, 0.043872, 0.045067, 0.046244, 0.047404, 0.048548, 0.049678, 0.050794, 0.051898, 0.052990,
    0.054073, 0.055146, 0.056212, 0.057270, 0.058322, 0.059369, 0.060411, 0.061449, 0.062483, 0.063514,
    0.064542, 0.065567, 0.066589, 0.067608, 0.068623, 0.069634, 0.070641, 0.071643, 0.072638, 0.073627,
    0.074608, 0.075579, 0.076541, 0.077490, 0.078426, 0.079348, 0.080254, 0.081141, 0.082010, 0.082857,
    0.083681, 0.084480, 0.085254, 0.085999, 0.086715, 0.087400, 0.088052, 0.088669, 0.089251, 0.089795,
    0.090301, 0.090767, 0.091193, 0.091576, 0.091917, 0.092214, 0.092467, 0.092674, 0.092836, 0.092952,
    0.093022, 0.093046, 0.093023, 0.092953, 0.092838, 0.092676, 0.092468, 0.092214, 0.091915, 0.091572,
    0.091184, 0.090753, 0.090279, 0.089763, 0.089205, 0.088606, 0.087968, 0.087289, 0.086573, 0.085818,
    0.085026, 0.084198, 0.083333, 0.082434, 0.081499, 0.080531, 0.079529, 0.078494, 0.077426, 0.076326,
    0.075193, 0.074029, 0.072834, 0.071607, 0.070348, 0.069059, 0.067739, 0.066387, 0.065005, 0.063591,
    0.062146, 0.060670, 0.059163, 0.057624, 0.056054, 0.054452, 0.052818, 0.051153, 0.049456, 0.047727,
    0.045966, 0.044174, 0.042349, 0.040493, 0.038606, 0.036687, 0.034737, 0.032757, 0.030746, 0.028705,
    0.026634, 0.024535, 0.022406, 0.020250, 0.018067, 0.015856, 0.013620, 0.011359, 0.009073, 0.006764,
    0.004432, 0.002078, -0.000297, -0.002691, -0.005105, -0.007537, -0.009986, -0.012451, -0.014931, -0.017426,
    -0.019933, -0.022454, -0.024986, -0.027529, -0.030081, -0.032643, -0.035212, -0.037790, -0.040373, -0.042963,
    -0.045558, -0.048158, -0.050761, -0.053367, -0.055977, -0.058588, -0.061200, -0.063814, -0.066427, -0.069041,
    -0.071653, -0.074264, -0.076872, -0.079478, -0.082080, -0.084678, -0.087272, -0.089860, -0.092441, -0.095015,
    -0.097581, -0.100138, -0.102685, -0.105221, -0.107744, -0.110255, -0.112750, -0.115230, -0.117692, -0.120136,
    -0.122560, -0.124961, -0.127340, -0.129693, -0.132019, -0.134317, -0.136585, -0.138820, -0.141022, -0.143187,
    -0.145314, -0.147402, -0.149448, -0.151449, -0.153405, -0.155313, -0.157171, -0.158977, -0.160730, -0.162427,
    -0.164067, -0.165647, -0.167167, -0.168624, -0.170016, -0.171343, -0.172602, -0.173792, -0.174912, -0.175961,
    -0.176936, -0.177838, -0.178664, -0.179414, -0.180087, -0.180682, -0.181197, -0.181633, -0.181989, -0.182263,
    -0.182456, -0.182566, -0.182593, -0.182536, -0.182395, -0.182169, -0.181858, -0.181461, -0.180978, -0.180407,
    -0.179750, -0.179004, -0.178169, -0.177245, -0.176231, -0.175125, -0.173928, -0.172638, -0.171255, -0.169777,
    -0.168203, -0.166532, -0.164763, -0.162895, -0.160926, -0.158855, -0.156681, -0.154401, -0.152015, -0.149521,
    -0.146916, -0.144201, -0.141372, -0.138428, -0.135367, -0.132188, -0.128889, -0.125468, -0.121923, -0.118253,
    -0.114456, -0.110530, -0.106474, -0.102286, -0.097964, -0.093508, -0.088916, -0.084187, -0.079319, -0.074312,
    -0.069164, -0.063874, -0.058442, -0.052867, -0.047148, -0.041285, -0.035276, -0.029122, -0.022823, -0.016377,
    -0.009786, -0.003048, 0.003836, 0.010867, 0.018044, 0.025367, 0.032837, 0.040452, 0.048213, 0.056120,
    0.064172, 0.072370, 0.080713, 0.089201, 0.097833, 0.106611, 0.115533, 0.124600, 0.133811, 0.143166,
    0.152666, 0.162311, 0.172100, 0.182033, 0.192111, 0.202333, 0.212700, 0.223211, 0.233866, 0.244665,
    0.255608, 0.266694, 0.277924, 0.289295, 0.300808, 0.312462, 0.324255, 0.336185, 0.348252, 0.360453,
    0.372786, 0.385249, 0.397837, 0.410549, 0.423379, 0.436325, 0.449380, 0.462541, 0.475801, 0.489154,
    0.502594, 0.516112, 0.529702, 0.543354, 0.557061, 0.570811, 0.584595, 0.598402, 0.612221, 0.626040,
    0.639845, 0.653625, 0.667364, 0.681050, 0.694667, 0.708200, 0.721633, 0.734951, 0.748136, 0.761172,
    0.774043, 0.786730, 0.799216, 0.811483, 0.823513, 0.835289, 0.846793, 0.858006, 0.868910, 0.879489,
    0.889725, 0.899600, 0.909099, 0.918204, 0.926900, 0.935171, 0.943002, 0.950380, 0.957291, 0.963722,
    0.969661, 0.975097, 0.980019, 0.984419, 0.988288, 0.991618, 0.994403, 0.996637, 0.998317, 0.999439,
    1.000000, 1.000000, 0.999439, 0.998317, 0.996637, 0.994403, 0.991618, 0.988288, 0.984419, 0.980019,
    0.975097, 0.969661, 0.963722, 0.957291, 0.950380, 0.943002, 0.935171, 0.926900, 0.918204, 0.909099,
    0.899600, 0.889725, 0.879489, 0.868910, 0.858006, 0.846793, 0.835289, 0.823513, 0.811483, 0.799216,
    0.786730, 0.774043, 0.761172, 0.748136, 0.734951, 0.721633, 0.708200, 0.694667, 0.681050, 0.667364,
    0.653625, 0.639845, 0.626040, 0.612221, 0.598402, 0.584595, 0.570811, 0.557061, 0.543354, 0.529702,
    0.516112, 0.502594, 0.489154, 0.475801, 0.462541, 0.449380, 0.436325, 0.423379, 0.410549, 0.397837,
    0.385249, 0.372786, 0.360453, 0.348252, 0.336185, 0.324255, 0.312462, 0.300808, 0.289295, 0.277924,
    0.266694, 0.255608, 0.244665, 0.233866, 0.223211, 0.212700, 0.202333, 0.192111, 0.182033, 0.172100,
    0.162311, 0.152666, 0.143166, 0.133811, 0.124600, 0.115533, 0.106611, 0.097833, 0.089201, 0.080713,
    0.072370, 0.064172, 0.056120, 0.048213, 0.040452, 0.032837, 0.025367, 0.018044, 0.010867, 0.003836,
    -0.003048, -0.009786, -0.016377, -0.022823, -0.029122, -0.035276, -0.041285, -0.047148, -0.052867, -0.058442,
    -0.063874, -0.069164, -0.074312, -0.079319, -0.084187, -0.088916, -0.093508, -0.097964, -0.102286, -0.106474,
    -0.110530, -0.114456, -0.118253, -0.121923, -0.125468, -0.128889, -0.132188, -0.135367, -0.138428, -0.141372,
    -0.144201, -0.146916, -0.149521, -0.152015, -0.154401, -0.156681, -0.158855, -0.160926, -0.162895, -0.164763,
    -0.166532, -0.168203, -0.169777, -0.171255, -0.172638, -0.173928, -0.175125, -0.176231, -0.177245, -0.178169,
    -0.179004, -0.179750, -0.180407, -0.180978, -0.181461, -0.181858, -0.182169, -0.182395, -0.182536, -0.182593,
    -0.182566, -0.182456, -0.182263, -0.181989, -0.181633, -0.181197, -0.180682, -0.180087, -0.179414, -0.178664,
    -0.177838, -0.176936, -0.175961, -0.174912, -0.173792, -0.172602, -0.171343, -0.170016, -0.168624, -0.167167,
    -0.165647, -0.164067, -0.162427, -0.160730, -0.158977, -0.157171, -0.155313, -0.153405, -0.151449, -0.149448,
    -0.147402, -0.145314, -0.143187, -0.141022, -0.138820, -0.136585, -0.134317, -0.132019, -0.129693, -0.127340,
    -0.124961, -0.122560, -0.120136, -0.117692, -0.115230, -0.112750, -0.110255, -0.107744, -0.105221, -0.102685,
    -0.100138, -0.097581, -0.095015, -0.092441, -0.089860, -0.087272, -0.084678, -0.082080, -0.079478, -0.076872,
    -0.074264, -0.071653, -0.069041, -0.066427, -0.063814, -0.061200, -0.058588, -0.055977, -0.053367, -0.050761,
    -0.048158, -0.045558, -0.042963, -0.040373, -0.037790, -0.035212, -0.032643, -0.030081, -0.027529, -0.024986,
    -0.022454, -0.019933, -0.017426, -0.014931, -0.012451, -0.009986, -0.007537, -0.005105, -0.002691, -0.000297,
    0.002078, 0.004432, 0.006764, 0.009073, 0.011359, 0.013620, 0.015856, 0.018067, 0.020250, 0.022406,
    0.024535, 0.026634, 0.028705, 0.030746, 0.032757, 0.034737, 0.036687, 0.038606, 0.040493, 0.042349,
    0.044174, 0.045966, 0.047727, 0.049456, 0.051153, 0.052818, 0.054452, 0.056054, 0.057624, 0.059163,
    0.060670, 0.062146, 0.063591, 0.065005, 0.066387, 0.067739, 0.069059, 0.070348, 0.071607, 0.072834,
    0.074029, 0.075193, 0.076326, 0.077426, 0.078494, 0.079529, 0.080531, 0.081499, 0.082434, 0.083333,
    0.084198, 0.085026, 0.085818, 0.086573, 0.087289, 0.087968, 0.088606, 0.089205, 0.089763, 0.090279,
    0.090753, 0.091184, 0.091572, 0.091915, 0.092214, 0.092468, 0.092676, 0.092838, 0.092953, 0.093023,
    0.093046, 0.093022, 0.092952, 0.092836, 0.092674, 0.092467, 0.092214, 0.091917, 0.091576, 0.091193,
    0.090767, 0.090301, 0.089795, 0.089251, 0.088669, 0.088052, 0.087400, 0.086715, 0.085999, 0.085254,
    0.084480, 0.083681, 0.082857, 0.082010, 0.081141, 0.080254, 0.079348, 0.078426, 0.077490, 0.076541,
    0.075579, 0.074608, 0.073627, 0.072638, 0.071643, 0.070641, 0.069634, 0.068623, 0.067608, 0.066589,
    0.065567, 0.064542, 0.063514, 0.062483, 0.061449, 0.060411, 0.059369, 0.058322, 0.057270, 0.056212,
    0.055146, 0.054073, 0.052990, 0.051898, 0.050794, 0.049678, 0.048548, 0.047404, 0.046244, 0.045067,
    0.043872, 0.042658, 0.041424, 0.040170, 0.038894, 0.037595, 0.036274, 0.034930, 0.033562, 0.032171,
    0.030757, 0.029319, 0.027858, 0.026374, 0.024869, 0.023343, 0.021798, 0.020234, 0.018652, 0.017056,
    0.015445, 0.013822, 0.012190, 0.010549, 0.008903, 0.007254, 0.005604, 0.003955, 0.002311, 0.000673,
    -0.000954, -0.002570, -0.004171, -0.005755, -0.007318, -0.008859, -0.010376, -0.011865, -0.013325, -0.014754,
    -0.016149, -0.017509, -0.018833, -0.020120, -0.021367, -0.022574, -0.023741, -0.024866, -0.025950, -0.026993,
    -0.027995, -0.028956, -0.029877, -0.030759, -0.031602, -0.032410, -0.033182, -0.033921, -0.034629, -0.035307,
    -0.035959, -0.036586, -0.037191, -0.037777, -0.038346, -0.038902, -0.039446, -0.039982, -0.040512, -0.041040,
    -0.041568, -0.042099, -0.042635, -0.043178, -0.043732, -0.044298, -0.044878, -0.045474, -0.046087, -0.046720,
    -0.047372, -0.048045, -0.048740, -0.049456, -0.050195, -0.050955, -0.051737, -0.052539, -0.053361, -0.054201,
    -0.055058, -0.055930, -0.056815, -0.057711, -0.058615, -0.059525, -0.060438, -0.061351, -0.062260, -0.063164,
    -0.064057, -0.064937, -0.065801, -0.066645, -0.067465, -0.068258, -0.069020, -0.069749, -0.070442, -0.071093,
    -0.071702, -0.072266, -0.072780, -0.073244, -0.073655, -0.074011, -0.074310, -0.074552, -0.074733, -0.074855,
)


def _resample_burp_table(table, n_steps):
    """Resample a fixed reference SIGNED b1(tau) table (uniformly spaced
    over [0,1)) onto n_steps points via linear interpolation of the SIGNED
    profile -- NOT of separately-folded amp/phase, which would create
    spurious intermediate phase values right at a 0/180 sign-flip
    boundary."""
    table = np.asarray(table, dtype=float)
    tau_table = np.linspace(0.0, 1.0, len(table), endpoint=False)
    tau_out = np.linspace(0.0, 1.0, n_steps, endpoint=False)
    signed = np.interp(tau_out, tau_table, table)
    return np.abs(signed), np.where(signed < 0, 180.0, 0.0)


def generate_shape(shape_name, n_steps, burp_coeffs_A=None, burp_coeffs_B=None):
    """amp, phase_deg = generate_shape(shape_name, n_steps, burp_coeffs_A, burp_coeffs_B)

    EXACT duplicate of the pulse-programme family's generate_shape() (see
    module docstring) -- amp is peak-normalised to 1.0, phase_deg in
    degrees. shape_name: 'GAUSSIAN', 'SINC', 'EBURP2'/'REBURP' (fixed
    Bruker-sourced tables, no coefficients needed), or 'EBURP1'/'BURP'
    (needs burp_coeffs_A -- Geen, H. & Freeman, R., J. Magn. Reson. 93,
    93-141 (1991))."""
    shape_name = shape_name.upper()

    if shape_name == 'GAUSSIAN':
        tau = np.linspace(-2.5, 2.5, n_steps)
        amp = np.exp(-0.5*tau**2)
        phase = np.zeros(n_steps)

    elif shape_name == 'SINC':
        tau = np.linspace(-5.0, 5.0, n_steps)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = np.where(tau == 0, 1.0, np.sin(np.pi*tau)/(np.pi*tau))
        window = 0.54 + 0.46*np.cos(np.pi*tau/5.0)
        raw = raw*window
        amp = np.abs(raw)
        phase = np.where(raw < 0, 180.0, 0.0)

    elif shape_name == 'EBURP2':
        amp, phase = _resample_burp_table(_EBURP2_TABLE_B1, n_steps)

    elif shape_name == 'REBURP':
        amp, phase = _resample_burp_table(_REBURP_TABLE_B1, n_steps)

    elif shape_name in ('EBURP1', 'BURP'):
        if not burp_coeffs_A:
            raise ValueError(
                "shape '{0}' requested but no Fourier A-coefficients were "
                "supplied. EBURP1 is defined in Geen, H. & Freeman, R., "
                "J. Magn. Reson. 93, 93-141 (1991) as a truncated Fourier "
                "series -- enter coefficients, or use 'GAUSSIAN' / 'SINC' "
                "/ 'EBURP2' / 'REBURP' instead, which need no coefficients."
                .format(shape_name))
        tau = np.linspace(0.0, 1.0, n_steps, endpoint=False)
        b1 = np.full(n_steps, burp_coeffs_A[0], dtype=float)
        for n in range(1, max(len(burp_coeffs_A), len(burp_coeffs_B))):
            An = burp_coeffs_A[n] if n < len(burp_coeffs_A) else 0.0
            Bn = burp_coeffs_B[n] if n < len(burp_coeffs_B) else 0.0
            b1 = b1 + An*np.cos(2*np.pi*n*tau) + Bn*np.sin(2*np.pi*n*tau)
        amp = np.abs(b1)
        phase = np.where(b1 < 0, 180.0, 0.0)

    else:
        raise ValueError("Unknown on-the-fly shape '{0}'. Known: GAUSSIAN, "
                          "SINC, EBURP2, REBURP, EBURP1/BURP (needs "
                          "BurpCoeffsA/B).".format(shape_name))

    peak = np.max(np.abs(amp))
    if peak > 0:
        amp = amp/peak
    return amp, phase


def hard_pulse_to_b1_hz(relative_amplitude, hard_pulse_width_us, hard_pulse_amplitude,
                         lp_max_fraction=1.0):
    """b1_hz = hard_pulse_to_b1_hz(relative_amplitude, hard_pulse_width_us,
                                    hard_pulse_amplitude, lp_max_fraction=1.0)

    Convert a relative TX amplitude (0..1, e.g. RFAsh0) into a physical B1
    field strength, expressed as a nutation frequency in Hz (gamma*B1 /
    2*pi -- i.e. 1/(4*t) is the nutation frequency of a pulse that gives a
    90 degree rotation in time t), using a calibrated hard 90 pulse as the
    reference point. This is the SAME physics used to derive RFAsh0's
    default in LWG_Selective-PulseAcquire_H.py.

    relative_amplitude: the TX scale you want converted (e.g. RFAsh0,
        0.0..1.0 of LP max).
    hard_pulse_width_us: width [us] of your calibrated hard 90 pulse
        (P90/P1Hard) at hard_pulse_amplitude, on the SAME port your hard-
        pulse calibration was done on (normally HP).
    hard_pulse_amplitude: relative TX scale (0.0..1.0) of that hard 90
        pulse (RFA0).
    lp_max_fraction: if relative_amplitude is on a DIFFERENT port than the
        hard-pulse reference (e.g. RFAsh0 is LP-relative but your hard 90
        was calibrated on HP), pass your measured LP-vs-HP max-power ratio
        (LPMaxFraction) here to re-reference it. Leave at 1.0 if both are
        on the same port/scale.
    """
    if hard_pulse_width_us <= 0 or hard_pulse_amplitude <= 0:
        raise ValueError("hard_pulse_to_b1_hz: hard_pulse_width_us and "
                          "hard_pulse_amplitude must both be > 0.")
    hard_pulse_nutation_hz = 1.0e6 / (4.0 * hard_pulse_width_us)  # 90deg = 1/4 cycle
    hp_equivalent_amplitude = relative_amplitude * lp_max_fraction
    return hard_pulse_nutation_hz * (hp_equivalent_amplitude / hard_pulse_amplitude)



def _rotate_many(M, bx, by, bz, dt):
    """Rotate every isochromat in ``M`` (``(..., 3)``) about its own
    effective field ``(bx, by, bz)`` (Hz, broadcastable to ``M[..., 0]``)
    for ``dt`` seconds -- :func:`_rotate` of the original, vectorised."""
    mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    safe = np.where(mag > 0, mag, 1.0)
    kx, ky, kz = bx / safe, by / safe, bz / safe
    angle = np.where(mag > 0, 2.0 * np.pi * mag * dt, 0.0)
    c, s = np.cos(angle), np.sin(angle)
    mx, my, mz = M[..., 0], M[..., 1], M[..., 2]
    dot = kx * mx + ky * my + kz * mz
    cross_x = ky * mz - kz * my
    cross_y = kz * mx - kx * mz
    cross_z = kx * my - ky * mx
    out = np.empty_like(M)
    out[..., 0] = mx * c + cross_x * s + kx * dot * (1.0 - c)
    out[..., 1] = my * c + cross_y * s + ky * dot * (1.0 - c)
    out[..., 2] = mz * c + cross_z * s + kz * dot * (1.0 - c)
    return out


# ============================================================
# Adiabatic (frequency-swept) shapes
# ============================================================

#: The adiabatic shapes :func:`simulate_pulse` knows, with their
#: parameters' defaults.
#:
#: * ``HS`` -- hyperbolic secant, HSn (Silver, Joseph & Hoult, J. Magn.
#:   Reson. 59, 347 (1984); HSn: Tannus & Garwood, J. Magn. Reson. A 120,
#:   133 (1996)). Amplitude sech(beta tau^n); the frequency follows the
#:   integral of amplitude squared (offset-independent adiabaticity),
#:   which for n = 1 is the classic (SW/2) tanh(beta tau)/tanh(beta).
#:   beta = 5.3 truncates the amplitude at 1% (sech 5.3 = 0.01).
#: * ``WURST`` -- WURST-n (Kupce & Freeman, J. Magn. Reson. A 115, 273
#:   (1995)). Amplitude 1 - |sin(pi tau / 2)|^n, linear sweep.
#: * ``TANH_TAN`` -- tanh/tan adiabatic full passage (Hwang, van Zijl &
#:   Garwood, J. Magn. Reson. 133, 200 (1998)). Amplitude
#:   tanh(xi (1 - |tau|)), frequency (SW/2) tan(kappa tau)/tan(kappa);
#:   xi = 10, tan(kappa) = 20 as in that paper. Most of its frequency
#:   excursion happens in the outer ~10% of the pulse, while the
#:   amplitude ramps, so its inverted band is a small fraction of the
#:   sweep unless the pulse is long -- see :func:`adiabaticity`.
#:
#: tau runs from -1 to +1 across the pulse; the frequency sweeps from
#: -SW/2 to +SW/2.
ADIABATIC_SHAPES = {
    "HS": {"beta": 5.3, "n": 1},
    "WURST": {"n": 20},
    "TANH_TAN": {"xi": 10.0, "tan_kappa": 20.0},
}


#: Largest phase advance per time step (turns) at the edge of a sweep that
#: auto_time_step_us allows -- see the module notes for the measurement.
MAX_TURNS_PER_STEP = 0.05

#: Time steps auto_time_step_us picks from (us): the hardware's 1 us, then
#: round fractions of it.
_NICE_STEPS_US = (1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.025, 0.02, 0.01, 0.005, 0.0025, 0.002,
                  0.001)

#: A simulation longer than this many steps is refused (memory and time).
MAX_STEPS = 5_000_000


def auto_time_step_us(sweep_width_hz=None):
    """The time step (us) for a pulse: 1 us (the X-Pulse hardware's own
    resolution) unless a frequency sweep of ``sweep_width_hz`` needs a
    finer one, in which case the largest of 0.5, 0.25, 0.2, 0.1, ... us
    that keeps the phase ramp at the sweep's edges to at most
    :data:`MAX_TURNS_PER_STEP` of a turn per step."""
    if not sweep_width_hz or sweep_width_hz <= 0:
        return 1.0
    bound_us = MAX_TURNS_PER_STEP / (0.5 * float(sweep_width_hz)) * 1e6
    for step in _NICE_STEPS_US:
        if step <= bound_us * (1 + 1e-12):
            return step
    return bound_us


def n_steps_for(duration_us, time_step_us):
    """Steps of ``time_step_us`` in a pulse of ``duration_us`` (at least 1)."""
    if time_step_us is None or time_step_us <= 0:
        raise ValueError("The time step must be positive.")
    n = int(round(float(duration_us) / float(time_step_us)))
    if n > MAX_STEPS:
        raise ValueError(f"{duration_us:g} us at {time_step_us:g} us per step is {n:,} steps -- "
                         f"more than {MAX_STEPS:,}. Shorten the pulse or lengthen the step.")
    return max(1, n)


def is_adiabatic(shape):
    return str(shape).upper() in ADIABATIC_SHAPES


def adiabatic_shape(shape, n_steps, duration_us, sweep_width_hz, **params):
    """amp, phase_deg, freq_hz = adiabatic_shape(shape, n_steps, duration_us, sweep_width_hz, ...)

    ``shape`` one of :data:`ADIABATIC_SHAPES` ('HS', 'WURST',
    'TANH_TAN'); ``params`` override that shape's defaults there.
    ``amp`` is peak-normalised to 1; ``freq_hz`` is the RF frequency's
    offset from the carrier at each step (from -SW/2 to +SW/2), and
    ``phase_deg`` its running integral, 360 deg x sum(freq x dt) --
    which is how a frequency sweep is delivered as a phase-modulated
    shape, and how :func:`simulate_pulse` applies it. The phase is zero
    at the centre of the pulse."""
    key = str(shape).upper()
    if key not in ADIABATIC_SHAPES:
        raise ValueError(f"Unknown adiabatic shape {shape!r}; use one of {sorted(ADIABATIC_SHAPES)}.")
    unknown = set(params) - set(ADIABATIC_SHAPES[key])
    if unknown:
        raise ValueError(f"{key} takes {sorted(ADIABATIC_SHAPES[key])}, not {sorted(unknown)}.")
    p = {**ADIABATIC_SHAPES[key], **{k: v for k, v in params.items() if v is not None}}
    if sweep_width_hz is None or sweep_width_hz <= 0:
        raise ValueError(f"{key} is frequency-swept: give its sweep width (Hz).")
    n_steps = int(n_steps)
    dt = duration_us * 1e-6 / n_steps
    # The phase can only step by less than half a turn per point, i.e.
    # |freq| < 1/(2 dt) -- the hard limit. (Well below it, the ramp is
    # already coarse: see auto_time_step_us.)
    nyquist = 0.5 / dt
    if sweep_width_hz / 2.0 >= nyquist:
        raise ValueError(
            f"A {sweep_width_hz / 1e3:g} kHz sweep cannot be synthesised at one point per "
            f"{dt * 1e6:g} us: the frequency must stay within +/-{nyquist / 1e3:g} kHz. "
            "Use a finer time step (or leave it on auto), or reduce the sweep width.")
    tau = (np.arange(n_steps) + 0.5) / n_steps * 2.0 - 1.0          # step midpoints, -1..1
    half = sweep_width_hz / 2.0
    if key == "HS":
        beta, n = float(p["beta"]), float(p["n"])
        if beta <= 0 or n < 1:
            raise ValueError("HS needs beta > 0 and n >= 1.")
        amp = 1.0 / np.cosh(beta * np.abs(tau) ** n)
        # Offset-independent adiabaticity: the sweep follows the running
        # integral of amp^2 (exact tanh form for n = 1).
        weight = amp ** 2
        cumulative = np.cumsum(weight) - 0.5 * weight
        freq = half * (2.0 * cumulative / np.sum(weight) - 1.0)
    elif key == "WURST":
        n = float(p["n"])
        if n <= 0:
            raise ValueError("WURST needs n > 0.")
        amp = 1.0 - np.abs(np.sin(0.5 * np.pi * tau)) ** n
        freq = half * tau
    else:
        xi, tan_kappa = float(p["xi"]), float(p["tan_kappa"])
        if xi <= 0 or tan_kappa <= 0:
            raise ValueError("tanh/tan needs xi > 0 and tan(kappa) > 0.")
        kappa = np.arctan(tan_kappa)
        amp = np.tanh(xi * (1.0 - np.abs(tau)))
        freq = half * np.tan(kappa * tau) / tan_kappa
    peak = np.max(np.abs(amp))
    amp = amp / peak if peak > 0 else amp
    turns = np.cumsum(freq) * dt
    turns -= np.interp(0.0, tau, turns)
    return amp, np.rad2deg(2.0 * np.pi * turns), freq


#: Fraction of the sweep over which :func:`adiabaticity` reports the
#: smallest crossing Q (for information; the power is set from Q0).
ADIABATIC_BAND = 0.8


def adiabaticity(shape, duration_us, sweep_width_hz, peak_b1_hz, band=ADIABATIC_BAND,
                 time_step_us=None, **params):
    """Adiabaticity of a frequency-swept pulse.

    The sweep passes an isochromat at offset Omega at the time t where
    nu(t) = Omega; its adiabaticity factor there is
    Q(Omega) = 2 pi nu1(t)^2 / |d nu/dt| (nu1 the B1 nutation frequency,
    nu the RF frequency, both Hz). Q >> 1 is adiabatic; ~5 is a common
    working value.

    Returns a dict:

    * ``q0`` -- Q at the centre of the sweep: the standard design
      parameter (Kupce & Freeman's Q0 for WURST; for HS, Q is the same at
      every offset). :func:`peak_b1_for_adiabaticity` sets the power from it.
    * ``q_band`` -- the smallest Q over the central ``band`` (default 80%)
      of the sweep. Equal to ``q0`` for HS; lower for WURST (amplitude
      falling towards the edges) and much lower for tanh/tan (its tan
      sweep is fastest near the edges).
    * ``offset_hz``, ``q`` -- Q against offset across the sweep.

    **Q is a guide, not the answer.** It describes the passage through
    resonance only. With tanh/tan the amplitude ramps while the frequency
    is still sweeping fast, and a high B1 tilts the effective field too
    quickly there: in a 500 us, 100 kHz tanh/tan pulse, 20 kHz of B1
    inverts on resonance and 40 kHz does not (checked against a 100x
    finer time step -- it is the pulse, not the integration). The
    simulated profile (:func:`inverted_band_hz`) is what to trust.

    ``time_step_us``: as :func:`simulate_pulse` (default: auto)."""
    step = time_step_us or auto_time_step_us(sweep_width_hz)
    n_steps = max(3, n_steps_for(duration_us, step))
    amp, _phase, freq = adiabatic_shape(shape, n_steps, duration_us, sweep_width_hz, **params)
    dt = duration_us * 1e-6 / n_steps
    rate = np.abs(np.gradient(freq, dt))                          # Hz/s
    with np.errstate(divide="ignore"):
        q = 2.0 * np.pi * (peak_b1_hz * amp) ** 2 / rate
    inside = np.abs(freq) <= 0.5 * band * sweep_width_hz
    centre = int(np.argmin(np.abs(freq)))
    return {"q0": float(q[centre]), "q_band": float(np.min(q[inside])), "band": float(band),
            "offset_hz": freq, "q": q}


def peak_b1_for_adiabaticity(shape, duration_us, sweep_width_hz, q=5.0, time_step_us=None,
                             **params):
    """Peak B1 (Hz) giving adiabaticity ``q`` at the centre of the sweep
    (``q0`` of :func:`adiabaticity`) -- the adiabatic counterpart of
    :func:`peak_b1_for_flip_angle`. Q0 scales as B1 squared."""
    if q <= 0:
        raise ValueError("The adiabaticity factor must be positive.")
    q1 = adiabaticity(shape, duration_us, sweep_width_hz, 1.0, time_step_us=time_step_us,
                      **params)["q0"]
    return float(np.sqrt(q / q1))


def inverted_band_hz(sim, threshold=-0.9, power_index=None):
    """(low, high) offsets (Hz) of the widest contiguous band where Mz <
    ``threshold`` (default -0.9: 95% inversion), for the nominal power
    (scale nearest 1) unless ``power_index`` is given; None if nowhere."""
    if power_index is None:
        power_index = int(np.argmin(np.abs(np.asarray(sim["power_scales"]) - 1.0)))
    mz = np.asarray(sim["Mz"])[power_index]
    offsets = np.asarray(sim["offsets_hz"])
    inside = mz < threshold
    best, start = None, None
    for k, flag in enumerate(np.append(inside, False)):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            if best is None or k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    if best is None:
        return None
    return float(offsets[best[0]]), float(offsets[best[1] - 1])


def simulate_pulse(shape, duration_us, peak_b1_hz, offsets_hz,
                   burp_coeffs_A=None, burp_coeffs_B=None, power_scales=(1.0,),
                   sweep_width_hz=None, adiabatic_params=None, time_step_us=None):
    """Bloch-simulate a shaped pulse (no relaxation) over a range of
    resonance offsets, starting from thermal equilibrium (Mz = 1).

    shape: 'GAUSSIAN', 'SINC', 'EBURP2', 'REBURP', or 'EBURP1'/'BURP'
        (needs burp_coeffs_A/B); or an adiabatic shape, 'HS', 'WURST',
        'TANH_TAN' (needs sweep_width_hz; adiabatic_params overrides the
        shape's parameters -- see :data:`ADIABATIC_SHAPES`).
    duration_us: pulse duration [us].
    time_step_us: the shape's time resolution [us]. None (default): auto
        -- 1 us, the X-Pulse hardware's own synthesis, unless a frequency
        sweep needs finer (:func:`auto_time_step_us`; several-MHz
        adiabatic sweeps).
    peak_b1_hz: B1 at the shape's peak amplitude, as a nutation
        frequency [Hz]. See :func:`hard_pulse_to_b1_hz` and
        :func:`peak_b1_for_flip_angle`.
    offsets_hz: resonance offsets [Hz] to simulate.
    power_scales: multipliers applied to peak_b1_hz -- e.g.
        (0.8, 1.0, 1.2) to see a +/-20% power miscalibration.

    Returns a dict: 'time_us', 'amp_profile', 'phase_profile_deg',
    'offsets_hz', 'power_scales', 'peak_b1_hz', and 'Mx'/'My'/'Mz' each
    of shape (len(power_scales), len(offsets_hz)); for an adiabatic shape
    also 'freq_profile_hz', 'sweep_width_hz', 'adiabatic_params' and
    'adiabaticity' (:func:`adiabaticity` at the nominal power, without
    its per-step arrays) and 'max_turns_per_step' (the largest phase
    advance between steps, in turns -- above ~0.1 the ramp is too coarse
    to trust); and always 'time_step_us'.
    """
    if time_step_us is None or time_step_us <= 0:
        time_step_us = auto_time_step_us(sweep_width_hz if is_adiabatic(shape) else None)
    time_step_us = float(time_step_us)
    if duration_us < time_step_us:
        raise ValueError(f"simulate_pulse: duration_us too short (need >= {time_step_us:g} us).")
    n_steps = n_steps_for(duration_us, time_step_us)
    extra = {}
    if is_adiabatic(shape):
        params = dict(adiabatic_params or {})
        amp_profile, phase_profile, freq_profile = adiabatic_shape(
            shape, n_steps, duration_us, sweep_width_hz, **params)
        extra = {"freq_profile_hz": freq_profile, "sweep_width_hz": float(sweep_width_hz),
                 "adiabatic_params": {**ADIABATIC_SHAPES[shape.upper()], **params},
                 "adiabaticity": {k: v for k, v in adiabaticity(
                     shape, duration_us, sweep_width_hz, peak_b1_hz,
                     time_step_us=time_step_us, **params).items()
                     if k in ("q0", "q_band", "band")}}
    else:
        amp_profile, phase_profile = generate_shape(shape, n_steps, burp_coeffs_A, burp_coeffs_B)
    phase_rad = np.deg2rad(phase_profile)
    dt = float(duration_us) * 1.0e-6 / n_steps   # the time step (1 us unless finer is needed)
    if extra:
        # How coarse the phase ramp is: its largest advance between steps.
        # (Only for a sweep: an amplitude-modulated shape's phase jumps by
        # half a turn where its lobes change sign, which is not a ramp.)
        turns = np.abs(np.diff(np.asarray(phase_profile, dtype=float))) / 360.0 % 1.0
        extra["max_turns_per_step"] = float(np.max(np.minimum(turns, 1.0 - turns))) \
            if turns.size else 0.0

    offsets_hz = np.asarray(offsets_hz, dtype=float)
    power_scales = np.asarray(power_scales, dtype=float)

    M = np.zeros((len(power_scales), len(offsets_hz), 3))
    M[..., 2] = 1.0
    scale = power_scales[:, None] * peak_b1_hz
    offset = offsets_hz[None, :]
    for i in range(n_steps):
        bx = scale * amp_profile[i] * np.cos(phase_rad[i])
        by = scale * amp_profile[i] * np.sin(phase_rad[i])
        M = _rotate_many(M, bx, by, offset, dt)

    return {
        "time_us": (np.arange(n_steps) + 0.5) * dt * 1e6,
        "time_step_us": round(dt * 1e6, 12),
        "amp_profile": amp_profile,
        "phase_profile_deg": phase_profile,
        "offsets_hz": offsets_hz,
        "power_scales": power_scales,
        "peak_b1_hz": peak_b1_hz,
        "Mx": M[..., 0], "My": M[..., 1], "Mz": M[..., 2],
        "shape": shape.upper(),
        "duration_us": float(duration_us),
        **extra,
    }


def integration_factor(shape, n_steps=1000, burp_coeffs_A=None, burp_coeffs_B=None):
    """The shape's coherent integration factor, ``mean(b1)/max|b1|``
    with b1 signed -- the fraction of a rectangular pulse's rotation
    the shape gives at the same peak B1, on resonance. For the BURP
    tables this reproduces the Bruker shape files' own SHAPE_INTEGFAC
    (see the table comments above)."""
    amp, phase = generate_shape(shape, n_steps, burp_coeffs_A, burp_coeffs_B)
    signed = amp * np.cos(np.deg2rad(phase))
    return float(np.mean(signed) / np.max(np.abs(signed)))


def peak_b1_for_flip_angle(shape, duration_us, flip_angle_deg,
                           burp_coeffs_A=None, burp_coeffs_B=None):
    """Peak B1 [Hz] giving ``flip_angle_deg`` on resonance: a rotation of
    ``flip/360`` cycles, delivered by a shape whose average B1 is
    ``integration_factor`` times its peak, over ``duration_us``.

    Exact for an amplitude-modulated shape on resonance, which is what
    this is for: a sensible starting power. Band-selective shapes
    (EBURP2, REBURP) are designed for one particular flip, and their
    profile is only right at the power that gives it. An adiabatic
    shape has no flip angle: see :func:`peak_b1_for_adiabaticity`."""
    if is_adiabatic(shape):
        raise ValueError(f"{shape.upper()} is adiabatic: its effect does not scale with power the "
                         "way a flip angle does. Set its power by adiabaticity (Q) instead.")
    n_steps = max(1, int(round(duration_us)))
    factor = integration_factor(shape, n_steps, burp_coeffs_A, burp_coeffs_B)
    if factor == 0:
        raise ValueError(f"Shape {shape} has no net on-resonance rotation.")
    return abs((flip_angle_deg / 360.0) / (duration_us * 1.0e-6 * factor))


def default_offset_range_hz(duration_us, sweep_width_hz=None):
    """Roughly +/-2x the shape's naive time-bandwidth bandwidth
    (4/duration) -- the original ``simulate_selective_pulse`` default:
    a generous starting guess, not a precise value. For an adiabatic
    shape (``sweep_width_hz`` given): +/- the sweep width, so both edges
    of the inverted band are in view."""
    if sweep_width_hz:
        return (-float(sweep_width_hz), float(sweep_width_hz))
    approx_bw_hz = 4.0e6 / duration_us
    return (-2.0 * approx_bw_hz, 2.0 * approx_bw_hz)


def frequency_unit(max_abs_hz):
    """(scale, unit) to show frequencies up to ``max_abs_hz`` in: Hz below
    10 kHz, kHz below 1 MHz, MHz above -- so a several-MHz sweep reads
    "2 MHz" rather than "2000000 Hz" or a bare "1e6" axis multiplier."""
    m = abs(float(max_abs_hz or 0.0))
    if m >= 1e6:
        return 1e6, "MHz"
    if m >= 1e4:
        return 1e3, "kHz"
    return 1.0, "Hz"


#: The pulse's phase/frequency trace: Okabe–Ito vermilion, apart from
#: the black amplitude for every kind of colour vision.
_PHASE_COLOUR = "#D55E00"


def plot_pulse_simulation(sim, fig, title=None):
    """Draw a :func:`simulate_pulse` result onto ``fig``: pulse amplitude
    and phase against time on top, Mx/My/Mz against offset below (one
    set of curves per power scale, labelled when there is more than
    one). The original's ``plot_pulse_simulation``, drawing onto a
    supplied figure instead of creating one."""
    import matplotlib.cm as cm

    fig.clear()
    ax_shape, ax_prof = fig.subplots(2, 1)
    has_power_sweep = len(sim["power_scales"]) > 1

    ax_shape.plot(sim["time_us"], sim["amp_profile"], color='k', label='Amplitude (normalised)')
    ax_shape.set_xlabel("Time / μs")
    ax_shape.set_ylabel("Amplitude (normalised)")
    ax_shape.set_ylim(-0.05, 1.05)
    ax_phase = ax_shape.twinx()
    if "freq_profile_hz" in sim:
        # A frequency-swept pulse's phase runs over many turns and says
        # little; its frequency sweep is what defines it.
        freq = np.asarray(sim["freq_profile_hz"])
        f_scale, f_unit = (1e6, "MHz") if np.max(np.abs(freq)) >= 1e6 else (1e3, "kHz")
        ax_phase.plot(sim["time_us"], freq / f_scale, color=_PHASE_COLOUR,
                      linestyle='--', label=f'Frequency [{f_unit}]')
        ax_phase.set_ylabel(f"RF offset / {f_unit}", color=_PHASE_COLOUR)
    else:
        ax_phase.plot(sim["time_us"], sim["phase_profile_deg"], color=_PHASE_COLOUR,
                      linestyle='--', label='Phase [deg]')
        ax_phase.set_ylabel("Phase / deg", color=_PHASE_COLOUR)
    ax_phase.tick_params(axis='y', labelcolor=_PHASE_COLOUR)
    ax_shape.set_title("Pulse shape")

    # Colour-blind friendly for the number of power levels (an ordered
    # sweep); Mz/Mx/My are told apart by line style.
    from ..plotting.accessibility import default_series_colours
    colors = (default_series_colours(len(sim["power_scales"]), ordered=True)
              if has_power_sweep else ["#0072B2"])     # Okabe–Ito blue
    o_scale, o_unit = frequency_unit(np.max(np.abs(sim["offsets_hz"])) if len(sim["offsets_hz"])
                                     else 0.0)
    offsets = np.asarray(sim["offsets_hz"], dtype=float) / o_scale
    for pi, pscale in enumerate(sim["power_scales"]):
        label_suffix = " ({0:.0%} power)".format(pscale) if has_power_sweep else ""
        ax_prof.plot(offsets, sim["Mz"][pi], color=colors[pi],
                     linestyle='-', label='Mz' + label_suffix)
        ax_prof.plot(offsets, sim["Mx"][pi], color=colors[pi],
                     linestyle='--', alpha=0.7, label='Mx' + label_suffix)
        ax_prof.plot(offsets, sim["My"][pi], color=colors[pi],
                     linestyle=':', alpha=0.7, label='My' + label_suffix)
    ax_prof.axhline(0, color='gray', linewidth=0.5)
    ax_prof.set_xlabel(f"Offset / {o_unit}")
    ax_prof.set_ylabel("Magnetisation (M0=1)")
    ax_prof.set_ylim(-1.05, 1.05)
    ax_prof.set_title("Excitation profile (no relaxation)")
    ax_prof.legend(fontsize=8, ncol=3 if has_power_sweep else 1)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return ax_shape, ax_prof

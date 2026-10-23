# Here, we deal with 87Rb with a CQ of 3.3 MHz and asymmetry of 0.7
# The second-order quadrupolar effect is considered (second number after "quadrupole" tag in spinsys)
# Note also that the detect operator is I1c: the I^{+} operator for the central transition only

spinsys {
  channels 87Rb
  nuclei 87Rb
  quadrupole 1 2 3.3e6 0.7 0 0 0
}

par {
   method           direct
   gamma_angles     1
   spin_rate        0
   crystal_file     zcw4180
   proton_frequency 400e6
   start_operator   I1x
   detect_operator  I1c
   np               512
   sw               5e4
   verbose          1
   variable tsw     1e6/sw
}

proc pulseq {} {
   global par
   acq_block {
      delay $par(tsw)
   }
}

# Below, we run experiments both in static and under MAS

proc main {} {
   global par

   # Static
   set f [fsimpson]
   faddlb $f 500 0
   fzerofill $f 4096
   fft $f
   fsave $f static_simulation.csdf
   #fset $f Static

   # MAS   
   set par(spin_rate) 10000
   set par(gamma_angles) 9
   set g [fsimpson]
   faddlb $g 250 0
   fzerofill $g 4096
   fft $g
   #fset $g MAS
   fsave $g mas_simulation.csdf

   #fsave [list $f $g] $par(name).csdf
}


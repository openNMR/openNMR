##############################
#### Define a spin system ####
##############################

# Focus on 31P nucleus with:
# 	 -100 kHz d_iso
# 	 +400 kHz large_delta (Haeberlen)
# 	 0.0 anisotropy (Haeberlen)
# 	 0, 0, 0 alpha, beta, gamma angles

spinsys {	
	channels 31P
	nuclei 31P
	shift 1 -100.0e3 400.0e3 0.0 0 0 0
}

######################
##### Parameters #####
######################

# verbose controls the output, and consists of eight numbers (either a 1 for yes or 0 for no): 
# 	(1) spin system
# 	(2) simulation progress
# 	(3) simulation info
# 	(4) start and detect operators
# 	(5) powder angles
# 	(6) rf averaging profile
# 	(7) acquisition block
# 	(8) averaging file
#
# We will run the simulation assuming an extenral field of 500 MHz (set by proton_frequency)
# The MAS rate is 100 kHz
# crystal_file: dictates how many orientations to sample and the type of sampling ("rep" are generated using the REPULSION algorithm; zcw are using another algorithm)
# gamma_angles: how many carousels to consider (ensure this is more than the number of sidebands)
# start_operator: where your magnetisation is before beginning
# detect_operator: I_{+} - i.e., I_x + i*I_y for spin 1 in this case
# sw: sweep width (typically easiest to be the spin_rate * gamma_angles)
# variable xx: variable names and definitions
# 		here, this is the time step that the FID is recorded in
# np: number of points to collect in the FID

par {
	method           	gcompute
	verbose          	0100
	
	proton_frequency 	500.0e6
	spin_rate        	100000
	
	crystal_file     	rep256
	gamma_angles     	60
		
	start_operator   	I1z
	detect_operator  	I1p
	
	sw			spin_rate*gamma_angles
	variable tsw		1.0e6/sw
	np               	8192*6
}

######################
### Pulse sequence ###
######################
#
# Now define the pulse sequence to be applied
# Here, we set the global system parameters to define what will happen
# Firstly, we reset the spin system back to the starting_operator (here, I_z)
# Next, we apply a shaped pulse with pulse_shape:
# 	pulse_shape $duration $shape1 $shape2 (for a two-spin system)
# 	In the example here, we have the "WURST" shape, with duration "wurst" and a shape "wavelist_wurst" (defined below)
# Having performed the shaped pulse, we apply an ideal pulse (pulseid):
# 	pulseid $duration $rf_amplitude $phase
# 	NOTE: pulseid has no true duration - the $duration is simply used to compute the flip angle(s)
# 	So here, we have a pulse of 500 kHz strength aimed along "-y"
# 	The 0.5 us duration: this means that the flip angle will be 0.5e-06 * 500e+03 * 2*pi = pi/2
# 	(In other words, the flip angle is = duration * rf_amplitude * 2pi, the 2pi given because all SIMPSON frequencies are in Hz.)
# Then we actually acquire with acq_block. Here, FID points are separated by the inverse of the dwell time, and the propagator is obtained using whatever instruction is contained in acq_block
# Here, it is just a delay
# In other words, this corresponds to an adiabatic pulse to spin-lock onto -z, then a pulse along -y, which will rotate the magnetisation back into the xy plane (specifically along -x...I think)
 

proc pulseq {} {
	global par 
	
	reset
	pulse_shaped $par(wurst) $par(wavelist_wurst)
	pulseid 0.5 500.0e3 -y

	acq_block { 
		delay $par(tsw)
	}
}

###########################
###### Pulse section ######
###########################

proc main {} {
	global par 

	#DEFINE power level in Hz
	set par(amp_max_wurst)		400.0e3

	#DEFINE WURST pulse length in us (FLOAT!)						
	set par(wurst) 				10.0
																								
	#DEFINE stepsize of WURST pulse in us						
	set par(ss_wurst)			0.05				
												
	#DEFINE sweep width in MHz (FLOAT!) and direction 				
	set par(sweep_wurst)		1.0
	set par(direction_wurst)	-1						
												
	#DEFINE slope of WURST profile							
	set par(n_wurst)			10
									
	#Destroys internal representations of all shaped pulses and frees the allocated memory
	free_all_shapes

	#Create internal representation of pulses
	set par(wavelist_wurst) 	[list2shape [wurstshape]]							
									
	########################
	###### Simulation ######
	########################
	
	#Open outputfile
	set roto		[expr round($par(spin_rate)/1000)]
	set wurstpwl 	[expr round($par(amp_max_wurst)/1000)]
	set name "inv_pow_$par(wurst)us_W$par(n_wurst)_$par(sweep_wurst)MHz_rot${roto}kHz_rf${wurstpwl}kHz"	
			
	# Start simulation
	set f [fsimpson]
	faddlb $f 30 0
	fzerofill $f [expr 4*$par(np)]
	fft $f
	fsave $f ${name}.spe
	fsave $f ${name}.dat -xreim
	save_shape 0 ${name}.shape
	funload $f		
}


proc wurstshape {} {
        global par

        #Definition of variables
        # This defines pi/2
        set pi2 [expr (4*atan(1))/2]

        #Amplitude
        # Below, we create a series of different parameters: ss, nsp and b
        # To call these, we can say ${name_of_parameter}
        # Here, ss is the step size taken in time (units of us)
        # nsp is the number of points (time steps) in the shape (remember from above that $par(wurst) is the pulse length in us
        # b is pi/pulse_length in rad/us
        set ss          $par(ss_wurst)
        set nsp         [expr $par(wurst)/$ss]
        set b           [expr (2*$pi2)/$par(wurst)]

        #Frequency
        # sweep_wurst is the sweep_width (in MHz)
        # So "k" below describes the sweep rate
        set k [expr ($par(sweep_wurst)/$par(wurst))]

        #Direction check and calculation of the values
        # The if and elseif statements just check whether we sweep the frequency low to high or high to low
        # Remember that direction_wurst = -1 is low to high and +1 is high to low...I think
        #
        # "amp" sets the amplitude of the pulse at each time step, whilst "ph" sets the phase
        # These are set according to the "standard" WURST pulse shape equations
        if {$par(direction_wurst) == -1} {
                for {set i 0} {$i <= $nsp} {incr i} {
                        set time        [expr $ss*$i]
                        set amp         [expr $par(amp_max_wurst)*(1-abs((cos($b*$time))**($par(n_wurst))))]
                        set ph          [expr fmod((0.5*$k*($time-$par(wurst)/2)**2),1)*360]

                        lappend shape [list $amp $ph]
                }
        } elseif {$par(direction_wurst) == 1} {
                set k [expr $k*(-1)]
                for {set i 0} {$i <= $nsp} {incr i} {
                        set time        [expr $ss*$i]
                        set amp         [expr $par(amp_max_wurst)*(1-(abs(sin($b*($time-$par(wurst)/2))))**($par(n_wurst)))]
                        set ph          [expr 360+fmod((0.5*$k*($time-$par(wurst)/2)**2),1)*360]

                        lappend shape [list $amp $ph]

                }
        } else {
                puts "Set direction"
        }
        return $shape
}

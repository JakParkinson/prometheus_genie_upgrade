#!/bin/awk -f
BEGIN {
    print "V 2000.1.2"
    print "TBEGIN ? ? ?"

    # DOM position
    x_center = 57.0
    y_center = 83.0
    z_center = -2001.93 + 1950

    # Muon parameters:
    x = x_center+7
    y = y_center
    z = z_center-5
    
    # Track event entry
    print "EM 0 1 0 0 0 0"
    
    # TR gen_num igen_num particle_type x y z zenith azimuth length energy time
    #   0      0       amu           x y z 90     0       30    5.1    0
    print "TR 0 0 amu-", x, y, z, "90 180 27.5 5 0"
    
    # Event end marker
    print "EE"
    print "TEND ? ? ?"
    print "END"
}


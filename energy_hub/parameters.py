#The parameters setting for the energy hub.
T = 24 # The first stage decision making
Tk = 96 # The second stage decision making
# The parameter is obtained from
eff_bat = 0.9
Emax = 10
Emin = 100
E0 = 50

# The parameter is obtained from
PBIC_max = 100
eff_bic = 0.95

# The maximal power exchange between the microgrid and utility grid
Pug_max = 100
Pug_min = 0

# The efficiency for the HVAC system
eff_HVAC = 0.9

#The heat, cooling and electrical demand curve
Electrical_load_profile_AC = [100,200,100,150,220,300,400,500,100,]
Electrical_load_profile_DC = [ ]
Thermal_load_profile = [ ]
Heat_load_profile = [ ]
PV_output_profile = [ ]

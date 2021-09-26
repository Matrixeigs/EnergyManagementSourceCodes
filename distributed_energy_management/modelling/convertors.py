# Convertor models for universal energy management system
# The models include the following types of convertors.
# 1) Bidirectional convertors
# 2) Unidirectional convertors
import distributed_energy_management.configuration.configuration_convertors as default_parameters
###################################1)Bidirectional convertors#################
BIC = \
    {
        "AREA": default_parameters.BIC["AREA"],
        "CAP": default_parameters.BIC["CAP"],
        "EFF_AC2DC": default_parameters.BIC["EFF_AC2DC"],
        "EFF_DC2AC": default_parameters.BIC["EFF_DC2AC"],
        "TIME_GENERATED": default_parameters.BIC["TIME_GENERATED"],
        "TIME_APPLIED": default_parameters.BIC["TIME_APPLIED"],
        "TIME_COMMANDED": default_parameters.BIC["TIME_COMMANDED"],
        "COMMAND_AC2DC":default_parameters.BIC["COMMAND_AC2DC"],
        "COMMAND_DC2AC":default_parameters.BIC["COMMAND_DC2AC"],
        "COMMAND_Q":default_parameters.BIC["COMMAND_DC2AC"],
    }

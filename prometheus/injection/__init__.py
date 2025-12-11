from ..particle import Particle, PropagatableParticle
from .registered_injectors import RegisteredInjectors
from .injection.injection import Injection
from .injection.LI_injection import LIInjection, injection_from_LI_output
from .injection.genie_injection import GENIEInjection, injection_from_GENIE_output
from .lepton_injector_utils import make_new_LI_injection
INJECTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: make_new_LI_injection
}

INJECTION_CONSTRUCTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: injection_from_LI_output
}

INJECTION_CONSTRUCTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: injection_from_LI_output,
    RegisteredInjectors.GENIE: lambda primary_file,prometheus_file, detector: injection_from_GENIE_output(primary_file,prometheus_file, detector.offset) ## adding genie with external file
}
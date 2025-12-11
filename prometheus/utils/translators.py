# A dict for converting PDG names to f2k names from
# https://www.zeuthen.desy.de/~steffenp/f2000/f2000_1.5.html#SEC26
PDG_to_f2k = {
    11:"e-", 
    -11:"e+",
    12:"nu_e",
    -12:"~nu_e",
    13:"mu-",
    -13:"mu+",
    14:"nu_mu",
    -14:"~nu_mu",
    22:"gamma",
    111:'pi0', # This is technically not defined but...
    211:'hadr',
    -211:'hadr',
    311:'k0', # This is technically not defined but...
    -311: "k0-", #lol
    321:'k+', # This is technically not defined but...
    -321:'k-', # This is technically not defined but...
    2212:'p+',
    -2212:'p-',
    3222: 'hadr', # just guessing idk
    411: "d+", #just guessing tbh
    -411: "d-",
    3112: "hadr", # its a  sigmaMinus so idk t2k
    3122: "hadr", # its a lambda baryon so uds quarks idk t2k
    -3122: "hadr",
    4122: "hadr", # its a LambdaCPlus
    421: "hadr", #D0 meson c,u_bar
    431:  "hadr",
    4212: "hadr", # Sigma Charmed Baryon?
    4222: "hadr", #Sigma Charmed baryon
    1000080160: "nuclues", #idk if this right name for t2k
    3212: "hadr", # Strange Baryon
    3224: "hadr", # Strange baryon
    130: "hadr", #strange kaon
    2000000101: "hadr", #proton
    9500: "smpPlus",
    -9500: "SMPMinus",
    213: "RhoPlus",
    -213: "RhoMinus",
    20213: "A1Plus",
    -20213: "A1Minus",
    100213: "Rho1465Plus",
    -100213: "Rho1465Minus"

}

f2k_to_PDG = {val:key for key, val in PDG_to_f2k.items()}

# mapping from https://github.com/icecube/LeptonInjector/blob/master/private/LeptonInjector/Particle.cxx
PDG_to_pstring = {
    0:"Unknown",
    11:"EMinus",
    12:"NuE",
    13:"MuMinus",
    14:"NuMu",
    15:"TauMinus",
    16:"NuTau",
    22: "photon", 
    -11:"EPlus",
    -12:"NuEBar",
    -13:"MuPlus",
    -14:"NuMuBar",
    -15:"TauPlus",
    -16:"NuTaBar",
    2212:"Proton", # I'm not sure that these are defined either but...
    -2212:"AntiProton", # I'm not sure that these are defined either but...
    2112:"Neutron",
    -2112:"AntiNeutron",
    211:'PiPlus', # I'm not sure that these are defined either but...
    -211:'PiMinus', # I'm not sure that these are defined either but...
    311:'KZero', # I'm not sure that these are defined either but...
    -311:"KZeroBar", # anti particel of K0
    321:"KPlus",
    -321:"KMinus",
    111:'PiZero', # I'm not sure that these are defined either but...
    -2000001006:"Hadrons",
    3222: "SigmaPlus", #idk
    411: "DPlus", #idk 
    3112: "SigmaMinus", #idk
    3122: "Lambda", #idk probably
    -3122: "LambdaBar", #lambda bar
    4122: "LambdaCPlus", #idk probably
    421: "DZero", #Its a meson c,u_bar
    431: "D_s+", # D_s+ meson
    4212: "SigmaCharm", # Sum(e±,μ±,τ±) Probably need to fix, coulds find the right strin
    1000080160: "O16", #not sure if right pstring name
    4222: "SigmaCharm", 
    3212: "SigmaZero", # Strange Baryon
    3224: "SigmaStarPlus", # Strange baryon
    130:  "StrangeKaon", # prob
    2000000101: "Proton",
    9500: "smpPlus",
    -9500: "SMPMinus",
    213: "RhoPlus",
    -213: "RhoMinus",
    20213: "A1Plus",
    -20213: "A1Minus",
    100213: "Rho1465Plus",
    -100213: "Rho1465Minus"

}

pstring_to_PDG = {val:key for key, val in PDG_to_pstring.items()}

# Mapping from https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/src/PROPOSAL/PROPOSAL/particle/Particle.h
# to https://www.zeuthen.desy.de/~steffenp/f2000/f2000_1.5.html#SEC26
int_type_to_str = {
    11: "e-",    # electron
    -11: "e+",   # positron
    # 12: "hadr",     # electron neutrino (
    # -12: "hadr",    # electron antineutrino
    # 13: "mupair",   # muon
    # -13: "mupair",  # antimuon
    # 14: "hadr",     # muon neutrino
    # -14: "hadr",    # muon antineutrino
    # 15: "hadr",     # tau 
    # -15: "hadr",    # antitau
    # 16: "hadr",     # t
    # -16: "hadr",    # tau antineutrino LOOK INTO
    22: "brems",    # photon
    111: "epair",   # neutral pion
    211: "hadr",    # positive pion
    -211: "hadr",   # negative pion
    311: "hadr",    # neutral kaon
    311: "hadr",    # neutral antikaon
    321: "hadr",    # positive kaon
    -321: "hadr",   # negative kaon
    2212: "hadr",   # proton
    -2212: "hadr",  # antiproton
    2112: "hadr",   # neutron
    -2112: "hadr",  # antineutron
    3222: "hadr", # Sigma + hadron 
    3112: "hadr", # Sigma - hadron
    411: "hadr", #probably idk its D+
    421: "hadr", #D0 meson  c,u_bar    -321:"hadr",
    
    3122: "hadr", #probabiliy idk its Lambda
    -3122: "hadr", #-Lambda bar
    4122: "hadr", # its Lambda c plus
    431: "hadr", #Ds+ meson 
    4212: "hadr", #Sigma Charm Baryon Come back to
    4222: "hadr", #Sigma Charm Baryon Come back to
    3212: "hadr", # Strange Baryon
    3224: "hadr", # Strange baryon
    130:   "hadr", # strange kaon?
    -2000001006: "hadr",  # generic hadrons
    1000000002: "brems",  # bremsstrahlung
    1000000003: "delta",  # delta ray
    1000000004: "epair",  # electron pair production
    1000000005: "hadr",   # photonuclear
    1000000006: "mupair", # muon pair production
    1000000007: "hadr",   # hadron production
    1000000008: "delta",  # delta ray (duplicate?)
    1000000011: "hadr", # decay
    1000000012: "epair", # positron annihlation with electron, no annihlation parameterisation in ppc, but same as general em cascade i think
    1000080160: "nucleus", # its O16 not sure about str name
    2000000101: "nucleus", # proton

    1000000018: "amu-" ### works for now....
}


str_to_int_type = {val: key for key, val in int_type_to_str.items()}

# Add a safe lookup function
def int_type_to_str_safe(int_type):
    return int_type_to_str.get(int_type, "unknown")

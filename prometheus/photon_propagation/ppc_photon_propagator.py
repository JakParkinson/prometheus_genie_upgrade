import numpy as np
import os
import subprocess

from .photon_propagator import PhotonPropagator
from .utils import should_propagate, parse_ppc
from ..lepton_propagation import LeptonPropagator, Loss
from ..detector import Detector
from ..particle import Particle
from ..utils import serialize_to_f2k, PDG_to_f2k


def ppc_sim(
    particle: Particle,
    det: Detector,
    lp: LeptonPropagator,
    ppc_config: dict
) -> None:
    """Simulate the propagation of a particle and of any photons resulting from
    the energy losses of this particle

    params
    ______
    particle: Particle to propagate
    det: Detector object to simulate within
    lp: Prometheus LeptonPropagator to imulate any charged leptons
    ppc_config: dictionary containg the configuration settings for the photon propagation
    """
    # TODO I think this could be factored out into a separate energy loss section
    # But that is not a now problem
    if abs(int(particle)) in [12, 14, 16]: # It's a neutrino
        return
    # TODO put this in config
    r_inice = det.outer_radius + 1000
    if abs(int(particle)) in [11, 13, 15]: # It's a charged lepton excluding electron
        print("propagating ", int(particle))
        lp.energy_losses(particle, det)
    # All of these we consider as point depositions
    elif abs(int(particle))==111: # It's a neutral pion
        # TODO handle this correctl by converting to photons after prop
        return
    elif abs(int(particle))==211 or abs(int(particle))==321: # It's a charged pion or electron
        print(f"Handling charged pion/kaon/electron {int(particle)}")
        if np.linalg.norm(particle.position-det.offset) <= r_inice:
            loss = Loss(int(particle), particle.e, particle.position, 0) ## no track length for pion
            particle.losses.append(loss)
    elif abs(int(particle))==311: # It's a neutral kaon
        print(f"Particle {int(particle)} is a neutral kaon, not propagating")
        # TODO handle this correctl by converting to photons after prop
        return
    elif  abs(int(particle)) in [2212, 2112, 321, 3222, 411, 421, 3112, 3122, 3212, 3223, 4122, 431, 4212, 4222, 130] or int(particle) == -2000001006 or int(particle) == 1000080160:  # All other hadrons plus O16
        print(f"returning hadron {int(particle)}")
        if np.linalg.norm(particle.position-det.offset) <= r_inice:
            loss = Loss(int(particle), particle.e, particle.position, 0) ## no track length for hadron
            particle.losses.append(loss)
        elif int(particle) == 22:  # It's a photon
            print(f"Handling photon {int(particle)}")
            if np.linalg.norm(particle.position-det.offset) <= r_inice:
                # Treat photon as a point-like energy deposition
                loss = Loss(int(particle), particle.e, particle.position, 0) # 0 track length
                particle.losses.append(loss)
                print(f"Photon deposited {particle.e:.2f} GeV at position {particle.position}")
            print(f"Photon {int(particle)} handled, not propagating further")
            return  # We don't need to propagate photons further
    elif int(particle) == 22:  # It's a photon
        print(f"Handling photon {int(particle)}")
        if np.linalg.norm(particle.position-det.offset) <= r_inice:
            # Treat photon as a point-like energy deposition
            loss = Loss(int(particle), particle.e, particle.position, 0) # 0 track length
            particle.losses.append(loss)
            print(f"Photon deposited {particle.e:.2f} GeV at position {particle.position}")
        return  # We don't need to propagate photons further
    elif int(particle) == 2000000101:
        print(f"Paritcle {int(particle)} is a genie construct , should have no photon yield")
        return
    else:
        # TODO make this into a custom error
        print(repr(particle))
        raise ValueError(f"Unrecognized particle: {int(particle)}")
    geo_tmpfile = f"{ppc_config['paths']['ppc_tmpdir']}/geo-f2k"
    ppc_tmpfile = f"{ppc_config['paths']['ppc_tmpdir']}/{ppc_config['paths']['ppc_tmpfile']}_{str(particle)}"
    f2k_tmpfile = f"{ppc_config['paths']['ppc_tmpdir']}/{ppc_config['paths']['f2k_tmpfile']}_{str(particle)}"
    command = f"{ppc_config['paths']['ppc_exe']} {ppc_config['simulation']['device']} < {f2k_tmpfile} > {ppc_tmpfile}"
    if ppc_config["simulation"]["supress_output"]:
        command += " 2>/dev/null"

    if not should_propagate(particle):
        return 
    serialize_to_f2k(particle, f2k_tmpfile)
    det.to_f2k(
        geo_tmpfile,
        serial_nos=[m.serial_no for m in det.modules]
    )
    tenv = os.environ.copy()
    tenv["PPCTABLESDIR"] = ppc_config["paths"]["ppc_tmpdir"]

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, env=tenv)
    process.wait()
    particle.hits = parse_ppc(ppc_tmpfile)
    for f in [geo_tmpfile, f2k_tmpfile, ppc_tmpfile]:
        os.remove(f)

    for child in particle.children:
        # TODO put this in config
        if child.e < 1: # GeV
            continue
        ppc_sim(child, det, lp, ppc_config)

class PPCPhotonPropagator(PhotonPropagator):
    """Interface for simulating energy losses and light propagation using PPC"""
    def propagate(self, particle: Particle) -> None:
        """Propagate input particle using PPC. Instead it modifies the 
        state of the input Particle. We should make this more consistent 
        but that is a problem for another day...

        params
        ______
        particle: Prometheus particle to propagate
        """
        return ppc_sim(particle, self.detector, self.lepton_propagator, self.config)

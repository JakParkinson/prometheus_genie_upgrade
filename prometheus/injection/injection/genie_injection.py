import numpy as np
import h5py as h5
import pandas as pd
from typing import Iterable

from ...particle import Particle, PropagatableParticle
from ..injection_event.injection_event import InjectionEvent
from ..interactions import Interactions
from .injection import Injection

import logging
logger = logging.getLogger(__name__)

class GENIEInjection(Injection):
    def __init__(self, events: Iterable[InjectionEvent]):
        if not all([isinstance(event, InjectionEvent) for event in events]):
            raise ValueError("You are trying to make GENIE Injection with non-GENIE events")
        super().__init__(events)
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        # genie specific properties ?
        return d

def injection_from_GENIE_output(primary_file: str, prometheus_file: str, detector_offset: np.ndarray) -> GENIEInjection:
    """Creates injection object from saved GENIE parquet files"""
    primary_set = pd.read_parquet(primary_file)
    prometheus_set = pd.read_parquet(prometheus_file)
    
    injection_events = []
    

    for idx in primary_set.index:
        if idx in prometheus_set.index:
            primary_row = primary_set.loc[idx]
            prometheus_row = prometheus_set.loc[idx]
            
   
            position = np.array(primary_row['position'])[0:3] # time messes with detector offset, i think time should be in mc_truth or something?
            adjusted_position = adjust_position(position, detector_offset)
            
            zenith = primary_row['theta']
            azimuth = primary_row['phi']
            
            initial_state = Particle(
                primary_row['pdg_code'],
                primary_row['e'],
                adjusted_position,
                np.array([
                    np.sin(zenith) * np.cos(azimuth),
                    np.sin(zenith) * np.sin(azimuth),
                    np.cos(zenith),
                ]),
                None
            )
            
            final_states = []
            
            final_pdg_codes = prometheus_row['pdg_code']
            final_energies = prometheus_row['e']
            final_thetas = prometheus_row['theta']
            final_phis = prometheus_row['phi']
            
            for i, pdg_code in enumerate(final_pdg_codes):
                # Create final state particle
                final_state = PropagatableParticle(
                    pdg_code,
                    final_energies[i],
                    adjusted_position,  # Same position as initial
                    np.array([
                        np.sin(final_thetas[i]) * np.cos(final_phis[i]),
                        np.sin(final_thetas[i]) * np.sin(final_phis[i]),
                        np.cos(final_thetas[i]),
                    ]),
                    None,
                    initial_state
                )
                final_states.append(final_state)
            
            interaction_type = primary_row['interaction']
            if interaction_type == 'CC':
                interaction = Interactions.CHARGED_CURRENT
            elif interaction_type == 'NC':
                interaction = Interactions.NEUTRAL_CURRENT
            else:
                interaction = Interactions.UNKNOWN
            
            event = InjectionEvent(
                initial_state,
                final_states,
                interaction,
                adjusted_position[0],
                adjusted_position[1],
                adjusted_position[2]
            )
            
            injection_events.append(event)
    
    return GENIEInjection(injection_events)


def adjust_position(position: np.ndarray, detector_offset: np.ndarray, max_distance: float = 10000.0) -> np.ndarray:
    """Adjust particle position to be within max_distance of detector center"""
    print('position:', position)
    print('detector offset', detector_offset)
    relative_position = position  + detector_offset
    distance = np.linalg.norm(relative_position)
    if distance > max_distance:
        scale_factor = max_distance / distance
        adjusted_position = detector_offset + relative_position * scale_factor
        print('returning ', adjusted_position)
        return adjusted_position
    print('relative_position, ', relative_position)
    return relative_position

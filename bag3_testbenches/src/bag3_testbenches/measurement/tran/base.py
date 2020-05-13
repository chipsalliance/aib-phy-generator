# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Sequence, Optional, Mapping, Type, List, Set, Union

import abc

from bag.design.module import Module
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict

from ...schematic.digital_tb_tran import bag3_testbenches__digital_tb_tran


class TranTB(TestbenchManager, abc.ABC):
    """This class provide utility methods useful for all transient simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_sim : float
            the total simulation time.

    subclasses can define the following optional entries:

    t_step : Optional[float]
        Optional.  The strobe period.  Defaults to no strobing.
    save_outputs : Sequence[str]
        Optional.  list of nets to save in simulation data file.
    tran_options : Mapping[str, Any]
        Optional.  transient simulation options dictionary.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def save_outputs(self) -> Sequence[str]:
        return self.specs.get('save_outputs', [])

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__digital_tb_tran

    @classmethod
    def sup_var_name(cls, sup_pin: str) -> str:
        return f'v_{sup_pin}'

    def get_bias_sources(self, sup_values: Mapping[str, Union[float, Mapping[str, float]]],
                         src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        """Save bias sources and pins into src_list and src_pins.

        Side effect: will add voltage variables in self.sim_params.
        """
        sim_params = self.sim_params
        env_params = self.env_params
        for sup_pin, sup_val in sup_values.items():
            if sup_pin in src_pins:
                raise ValueError(f'Cannot add bias source on pin {sup_pin}, already used.')

            var_name = self.sup_var_name(sup_pin)
            if sup_pin == 'VSS':
                if sup_val != 0:
                    raise ValueError('VSS must be 0 volts.')
            else:
                src_list.append(dict(type='vdc', lib='analogLib', value=var_name,
                                     conns=dict(PLUS=sup_pin, MINUS='VSS')))
                src_pins.add(sup_pin)
            if isinstance(sup_val, float) or isinstance(sup_val, int):
                sim_params[var_name] = float(sup_val)
            else:
                env_params[var_name] = dict(**sup_val)

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        t_step: Optional[float] = specs.get('t_step', None)
        tran_options: Mapping[str, Any] = specs.get('tran_options', {})

        tran_dict = dict(type='TRAN',
                         start=0.0,
                         stop='t_sim',
                         options=tran_options,
                         save_outputs=self.save_outputs,
                         )
        if t_step is not None:
            tran_dict['strobe'] = t_step

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [tran_dict]
        return netlist_info_from_dict(sim_setup)

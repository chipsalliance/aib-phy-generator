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

from typing import Dict, Any, List

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class aib_ams__aib_phase_interp(Module):
    """Module for library aib_ams cell aib_phase_interp.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_phase_interp.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pi_params='Phase interpolator parameters',
            dc_params='Delay cell parameters',
            num_core='Number of delay cells',
            export_dc_out='export delaycell output',
            inv_params='Inverter input buffer params',
            nand_params='Nand input buffer params',
            export_dc_in='Export the input of the delay cell'
        )

    def design(self, pi_params: Param, dc_params: Param, inv_params: List[Param],
               nand_params: List[Param], num_core: int, export_dc_out: bool, export_dc_in: bool,
               ) -> None:
        nbits = pi_params['nbits'] - 1
        self.instances['XDC'].design(**dc_params)
        self.instances['XINT'].design(**pi_params)
        if len(inv_params) == 1:
            self.instances['XINV'].design(**inv_params[0])
        else:
            self.array_instance('XINV', ['XINVL', 'XINVH'])
            self.instances['XINVL'].design(**inv_params[0])
            self.instances['XINVH'].design(**inv_params[1])
        if len(nand_params) == 1:
            self.instances['XNAND'].design(**nand_params[0])
        else:
            self.array_instance('XNAND', ['XNANDL', 'XNANDH'])
            self.instances['XNANDL'].design(**nand_params[0])
            self.instances['XNANDH'].design(**nand_params[1])
        # NOTE: always-on input on MSB to help with nonlinearity
        self.reconnect_instance('XINT', [
            (f'a_en<{nbits}:0>', f'VDD,sn<{nbits - 1}:0>'),
            (f'a_enb<{nbits}:0>', f'VSS,sp<{nbits - 1}:0>'),
            (f'b_en<{nbits}:0>', f'VSS,sp<{nbits - 1}:0>'),
            (f'b_enb<{nbits}:0>', f'VDD,sn<{nbits - 1}:0>'),
        ])
        if num_core > 1:
            co_sig = f'co_p<{num_core-2}:0>' if num_core > 2 else 'co_p'
            ci_sig = f'ci_p<{num_core-2}:0>' if num_core > 2 else 'ci_p'
            dc_conn_list = [
                ('bk1', 'VDD'),
                ('in_p', 'a_in_buf,' + co_sig),
                ('out_p', 'b_in,' + ci_sig),
                ('co_p', co_sig + ',mid'),
                ('ci_p', ci_sig + ',mid'),
            ]
            self.rename_instance('XDC', f'XDC<{num_core-1}:0>', conn_list=dc_conn_list)
        self.rename_pin('sp', f'sp<{nbits - 1}:0>')
        self.rename_pin('sn', f'sn<{nbits - 1}:0>')
        if export_dc_out:
            self.add_pin('b_in', 'output')
        if export_dc_in:
            self.add_pin('a_in_buf', 'output')

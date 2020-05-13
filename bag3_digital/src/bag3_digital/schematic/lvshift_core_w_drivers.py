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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType

# noinspection PyPep8Naming
class bag3_digital__lvshift_core_w_drivers(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'lvshift_core_w_drivers.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            core_params='schematic parameters for lvshift_core',
            buf_params='schematic parameters for output buffers',
            dual_output='True to export complementary outputs.',
            invert_out='True to export only the inverted output.',
            export_pins='Defaults to False.  True to export simulation pins.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dual_output=True,
            invert_out=False,
            export_pins=False,
        )

    def design(self, core_params: Param, buf_params: Param, dual_output: bool, invert_out: bool, export_pins: bool
               ) -> None:
        core = self.instances['XCORE']
        core.design(**core_params)

        if 'rst_casc' not in core.master.pins:
            # no reset
            self.remove_pin('rst_out')
            self.remove_pin('rst_outb')
            self.remove_pin('rst_casc')

        buf_nstage = len(buf_params['inv_params'])
        buf_invert = (buf_nstage % 2 == 1)
        if dual_output:
            if buf_invert:
                self.reconnect_instance('XBUFP', [('outb', 'outb')])
                self.reconnect_instance('XBUFN', [('outb', 'out')])
            self.instances['XBUFP'].design(dual_output=False, **buf_params)
            self.instances['XBUFN'].design(dual_output=False, **buf_params)
            self.remove_instance('XNC')
            if export_pins:
                self.add_pin('midn', TermType.output)
                self.add_pin('midp', TermType.output)
        else:
            self.remove_pin('out' if invert_out else 'outb')
            if buf_invert == invert_out:
                rm_inst = 'XBUFN'
                keep_inst = 'XBUFP'
                nc_name = 'midn'
            else:
                rm_inst = 'XBUFP'
                keep_inst = 'XBUFN'
                nc_name = 'midp'

            term_name = 'outb' if buf_invert else 'out'
            net_name = 'outb' if invert_out else 'out'
            self.remove_instance(rm_inst)
            self.reconnect_instance_terminal(keep_inst, term_name, net_name)
            if not export_pins:
                self.reconnect_instance_terminal('XNC', 'noConn', nc_name)
            else:
                self.add_pin('midn', TermType.output)
                self.add_pin('midp', TermType.output)
            self.instances[keep_inst].design(dual_output=False, **buf_params)

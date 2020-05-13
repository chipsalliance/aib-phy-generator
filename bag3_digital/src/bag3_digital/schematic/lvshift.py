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
class bag3_digital__lvshift(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'lvshift.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lev_params='parameters for level shifter',
            buf_params='parameters for input buffer.',
            dual_output='True to export complementary outputs.',
            invert_out='True to export only the inverted output.',
            export_pins='Defaults to False; set to True to export pins for simulation.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dual_output=True,
            invert_out=False,
            export_pins=False,
        )

    def design(self, lev_params: Param, buf_params: Param, dual_output: bool, invert_out: bool, export_pins: bool
               ) -> None:
        core = self.instances['XLEV']
        core.design(dual_output=dual_output, invert_out=invert_out, export_pins=export_pins, **lev_params)
        self.instances['XBUF'].design(dual_output=True, **buf_params)

        core_pins = core.master.pins
        if 'outb' not in core_pins:
            self.remove_pin('out' if invert_out else 'outb')
        if 'rst_casc' not in core_pins:
            # no reset
            self.remove_pin('rst_out')
            self.remove_pin('rst_outb')
            self.remove_pin('rst_casc')
        if export_pins:
            self.add_pin('inb_buf', TermType.output)
            self.add_pin('in_buf', TermType.output)
            self.add_pin('midp', TermType.output)
            self.add_pin('midn', TermType.output)
            self.reconnect_instance_terminal('XLEV', 'midn', 'midn')
            self.reconnect_instance_terminal('XLEV', 'midp', 'midp')

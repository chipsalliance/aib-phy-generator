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

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param, ImmutableList


# noinspection PyPep8Naming
class bag3_digital__inv_chain(Module):
    """Module for library bag3_digital cell inv_chain.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'inv_chain.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            inv_params='List of inverter parameters.',
            export_pins='True to export simulation pins.',
            dual_output='True to export complementary outputs  Ignored if only one stage.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            export_pins=False,
            dual_output=False,
        )

    def design(self, inv_params: ImmutableList[Param], export_pins: bool, dual_output: bool
               ) -> None:
        num = len(inv_params)
        if num < 1:
            raise ValueError('Cannot have 0 inverters.')
        if export_pins and dual_output:
            raise ValueError("oops! export_pins and dual_output cannot be True at the same time, "                             
                             "check inv_chain's schematic generator")
        if num == 1:
            self.instances['XINV'].design(**inv_params[0])
            self.remove_pin('out')
        else:
            if num == 2:
                pin_last2 = 'outb'
                if export_pins:
                    pin_last2 = 'mid'
                    self.rename_pin('outb', pin_last2)
                elif not dual_output:
                    self.remove_pin('outb')

                inst_term_list = [('XINV0', [('in', 'in'), ('out', pin_last2)]),
                                  ('XINV1', [('in', pin_last2), ('out', 'out')])]
            else:
                pin_last2 = 'outb' if num % 2 == 0 else 'out'
                if export_pins:
                    pin_last2 = f'mid<{num - 2}>'
                    self.rename_pin('outb' if num % 2 == 0 else 'out', f'mid<{num - 2}:0>')
                elif not dual_output:
                    self.remove_pin('outb' if num % 2 == 0 else 'out')

                inst_term_list = []
                for idx in range(num):
                    if idx == 0:
                        term = [('in', 'in'), ('out', 'mid<0>')]
                    elif idx == num - 1:
                        term = [('in', pin_last2), ('out', 'out' if num % 2 == 0 else 'outb')]
                    elif idx == num - 2:
                        term = [('in', f'mid<{idx - 1}>'), ('out', pin_last2)]
                    else:
                        term = [('in', f'mid<{idx - 1}>'), ('out', f'mid<{idx}>')]
                    inst_term_list.append((f'XINV{idx}', term))

            self.array_instance('XINV', inst_term_list=inst_term_list)
            for idx in range(num):
                self.instances[inst_term_list[idx][0]].design(**inv_params[idx])

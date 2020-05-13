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

from typing import Dict, Any, List, Optional

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__inv(Module):
    """Module for library bag3_digital cell inv.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'inv.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length in resolution units.',
            w_p='pmos width, in number of fins or resolution units.',
            w_n='nmos width, in number of fins or resolution units.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            seg='segments of transistors',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            p_in_gate_numbers='a List indicating input number of the gate',
            n_in_gate_numbers='a List indicating input number of the gate',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            p_in_gate_numbers=None,
            n_in_gate_numbers=None,
        )

    def design(self, seg: int, seg_p: int, seg_n: int, lch: int, w_p: int, w_n: int, th_p: str,
               th_n: str, stack_p: int, stack_n: int, p_in_gate_numbers: Optional[List[int]] = None,
               n_in_gate_numbers: Optional[List[int]] = None) -> None:
        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Cannot have negative number of segments.')

        self.instances['XN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=stack_n)
        self.instances['XP'].design(w=w_p, lch=lch, seg=seg_p, intent=th_p, stack=stack_p)

        self._reconnect_gate('XP', stack_p, p_in_gate_numbers, 'VSS')
        self._reconnect_gate('XN', stack_n, n_in_gate_numbers, 'VDD')

    def _reconnect_gate(self, inst_name: str, stack: int, idx_list: Optional[List[int]], sup: str
                        ) -> None:
        if stack > 1:
            g_term = f'g<{stack - 1}:0>'
            if idx_list:
                glist = [sup] * stack
                for i in idx_list:
                    glist[i] = 'in'
                self.reconnect_instance_terminal(inst_name, g_term, ','.join(glist))
            else:
                self.reconnect_instance_terminal(inst_name, g_term, 'in')
        else:
            self.reconnect_instance_terminal(inst_name, 'g', 'in')

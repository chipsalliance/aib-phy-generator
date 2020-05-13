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
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__nand(Module):
    """Module for library bag3_digital cell nand.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'nand.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            w_p='pmos width.',
            w_n='nmos width.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            num_in='number of inputs.',
            seg='segments of transistors',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            num_in=2,
        )

    def design(self, seg: int, seg_p: int, seg_n: int, lch: int, w_p: int, w_n: int, th_p: str,
               th_n: str, num_in: int, stack_p: int, stack_n: int) -> None:
        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Cannot have negative number of segments.')
        if num_in < 2:
            raise ValueError(f'num_in = {num_in} < 2')

        in_name = f'in<{num_in - 1}:0>'
        if num_in != 2:
            self.rename_pin('in<1:0>', in_name)

        # in net for pmos and nmos with stacking
        nin_list, pin_list = [], []
        for idx in range(num_in):
            nin_list = [f'in<{idx}>'] * stack_n + nin_list
            pin_list = [f'in<{idx}>'] * stack_p + pin_list
        nin_name = ','.join(nin_list)
        pin_name = ','.join(pin_list)

        pg_name = 'g' if stack_p == 1 else f'g<{stack_p - 1}:0>'

        self.instances['XP'].design(w=w_p, lch=lch, seg=seg_p, intent=th_p, stack=stack_p)
        self.rename_instance('XP', f'XP<{num_in - 1}:0>', [(pg_name, pin_name)])

        if stack_n & 1 and num_in > 2:
            # The layout requires the following for odd numbers of stacks and number of inputs > 2:
            # The pull-down network consists of input devices in series, where each of these input devices
            # consists of a segmented number of stacks
            ng_name = 'g' if stack_n == 1 else f'g<{stack_n - 1}:0>'
            self.instances['XN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=stack_n)
            term_list = []
            for i in range(num_in):
                conns = dict()
                conns[ng_name] = f'in<{i}>'
                if i != 0:
                    conns['s'] = f'nmid<{i - 1}>'
                if i != num_in - 1:
                    conns['d'] = f'nmid<{i}>'
                term_list.append(conns)
            self.array_instance('XN', [f'XN<{i}>' for i in range(num_in)], term_list)
        else:
            self.instances['XN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=num_in * stack_n)
            self.reconnect_instance_terminal('XN', f'g<{num_in * stack_n - 1}:0>', nin_name)

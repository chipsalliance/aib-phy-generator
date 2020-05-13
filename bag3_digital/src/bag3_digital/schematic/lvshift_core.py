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

from typing import Dict, Any, Mapping

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__lvshift_core(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'lvshift_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            seg_dict='dictionary of number of fingers.',
            w_dict='dictionary of number of fins.',
            intent_dict='dictionary of threshold types',
            in_upper='True to make the input connected to the upper transistor in the stack',
            has_rst='True to enable reset feature.',
            stack_p='PMOS number of stacks.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(in_upper=False, has_rst=False, stack_p=1)

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int],
               intent_dict: Mapping[str, str], in_upper: bool, has_rst: bool, stack_p: int) -> None:
        w_p = w_dict['pu']
        w_n = w_dict['pd']
        seg_p = seg_dict['pu']
        seg_n = seg_dict['pd']
        th_p = intent_dict['pch']
        th_n = intent_dict['nch']
        export_mid = (stack_p == 2)
        self.instances['XPP'].design(w=w_p, lch=lch, seg=seg_p, intent=th_p, stack=stack_p,
                                     export_mid=export_mid)
        self.instances['XPN'].design(w=w_p, lch=lch, seg=seg_p, intent=th_p, stack=stack_p,
                                     export_mid=export_mid)
        if stack_p == 1:
            self.reconnect_instance_terminal('XPP', 'g', 'outp')
            self.reconnect_instance_terminal('XPN', 'g', 'outn')

            self.remove_instance('XPRSTP')
            self.remove_instance('XPRSTN')
        else:
            if stack_p != 2:
                raise ValueError('stack_p has to be 2.')
            if not has_rst:
                raise ValueError('stack_p = 2 only allowed if has_rst = True')

            seg_prst = seg_dict['prst']
            self.instances['XPRSTP'].design(w=w_p, l=lch, nf=seg_prst, intent=th_p)
            self.instances['XPRSTN'].design(w=w_p, l=lch, nf=seg_prst, intent=th_p)

            mid_port = 'm' if seg_p == 1 else f'm<{seg_p-1}:0>'
            self.reconnect_instance_terminal('XPP', mid_port, 'midn')
            self.reconnect_instance_terminal('XPN', mid_port, 'midp')
            self.reconnect_instance_terminal('XPP', 'g<1:0>', 'inp,outp')
            self.reconnect_instance_terminal('XPN', 'g<1:0>', 'inn,outn')
            self.reconnect_instance_terminal('XPRSTP', 'S', 'midp')
            self.reconnect_instance_terminal('XPRSTN', 'S', 'midn')

        if has_rst:
            w_rst = w_dict['rst']
            seg_rst = seg_dict['rst']
            self.instances['XINP'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=2)
            self.instances['XINN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=2)
            self.instances['XRSTP'].design(w=w_rst, l=lch, nf=seg_rst, intent=th_n)
            self.instances['XRSTN'].design(w=w_rst, l=lch, nf=seg_rst, intent=th_n)
            if in_upper:
                self.reconnect_instance_terminal('XINP', 'g<1:0>', 'inp,rst_casc')
                self.reconnect_instance_terminal('XINN', 'g<1:0>', 'inn,rst_casc')
        else:
            self.remove_instance('XRSTP')
            self.remove_instance('XRSTN')
            self.remove_pin('rst_outp')
            self.remove_pin('rst_outn')
            self.remove_pin('rst_casc')

            self.instances['XINP'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=1)
            self.instances['XINN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=1)
            self.reconnect_instance_terminal('XINP', 'g', 'inp')
            self.reconnect_instance_terminal('XINN', 'g', 'inn')

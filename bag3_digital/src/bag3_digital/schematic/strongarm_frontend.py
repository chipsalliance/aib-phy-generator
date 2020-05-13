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
class bag3_digital__strongarm_frontend(Module):
    """Module for library bag3_digital cell strongarm_frontend.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'strongarm_frontend.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            seg_dict='transistor segments dictionary.',
            w_dict='transistor width dictionary.',
            th_dict='transistor threshold dictionary.',
            has_rstb='True to add rstb functionality.',
            has_bridge='True to add bridge switch.',
            stack_br='Number of stacks in bridge switch.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(has_rstb=False, has_bridge=False, stack_br=1)

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int],
               th_dict: Mapping[str, str], has_rstb: bool, has_bridge: bool, stack_br: int) -> None:

        for name in ['in', 'tail', 'nfb', 'pfb', 'swo', 'swm']:
            uname = name.upper()
            w = w_dict[name]
            nf = seg_dict[name]
            intent = th_dict[name]
            if name == 'tail':
                inst_name = 'XTAIL'

                if has_rstb:
                    self.instances[inst_name].design(lch=lch, w=w, seg=nf, intent=intent, stack=2)
                else:
                    self.instances[inst_name].design(lch=lch, w=w, seg=nf, intent=intent, stack=1)
                    self.reconnect_instance_terminal(inst_name, 'g', 'clk')
            elif name == 'swo':
                if has_rstb:
                    self.instances['XSWOP<1:0>'].design(l=lch, w=w, nf=nf, intent=intent)
                    self.instances['XSWON<1:0>'].design(l=lch, w=w, nf=nf, intent=intent)
                else:
                    self.rename_instance('XSWOP<1:0>', 'XSWOP')
                    self.rename_instance('XSWON<1:0>', 'XSWON')
                    self.instances['XSWOP'].design(l=lch, w=w, nf=nf, intent=intent)
                    self.instances['XSWON'].design(l=lch, w=w, nf=nf, intent=intent)
                    self.reconnect_instance('XSWOP', [('D', 'outp'), ('G', 'clk'),
                                                      ('S', 'VDD'), ('B', 'VDD')])
                    self.reconnect_instance('XSWON', [('D', 'outn'), ('G', 'clk'),
                                                      ('S', 'VDD'), ('B', 'VDD')])
            else:
                self.instances[f'X{uname}P'].design(l=lch, w=w, nf=nf, intent=intent)
                self.instances[f'X{uname}N'].design(l=lch, w=w, nf=nf, intent=intent)

        if has_bridge:
            w = w_dict['br']
            seg = seg_dict['br']
            intent = th_dict['br']
            self.instances['XBR'].design(lch=lch, w=w, seg=seg, intent=intent, stack=stack_br)
            if stack_br == 1:
                self.reconnect_instance_terminal('XBR', 'g', 'clk')
            else:
                self.reconnect_instance_terminal('XBR', f'g<{stack_br - 1}:0>', 'clk')
        else:
            self.remove_instance('XBR')

        if not has_rstb:
            self.remove_pin('rstb')

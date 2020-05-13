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
class bag3_digital__sr_latch_symmetric_core(Module):
    """Module for library bag3_digital cell sr_latch_symmetric_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sr_latch_symmetric_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length.',
            seg_dict='number of segments dictionary.',
            w_dict='widths dictionary.',
            th_dict='threshold dictionary.',
            has_rstb='True to add rstb functionality.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(has_rstb=False)

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int],
               th_dict: Mapping[str, str], has_rstb: bool) -> None:
        name_list = ['nfb', 'pfb', 'ps', 'nr']
        if has_rstb:
            name_list.append('pr')
        else:
            self.remove_pin('rstlb')
            self.remove_pin('rsthb')
            self.remove_instance('XPRP')
            self.remove_instance('XPRN')

        for name in name_list:
            uname = name.upper()
            w = w_dict[name]
            nf = seg_dict[name]
            intent = th_dict[name]
            pinst = self.instances[f'X{uname}P']
            ninst = self.instances[f'X{uname}N']
            if name[1:] == 'fb':
                pinst.design(lch=lch, w=w, seg=nf, intent=intent, stack=2)
                ninst.design(lch=lch, w=w, seg=nf, intent=intent, stack=2)
            else:
                pinst.design(l=lch, w=w, nf=nf, intent=intent)
                ninst.design(l=lch, w=w, nf=nf, intent=intent)

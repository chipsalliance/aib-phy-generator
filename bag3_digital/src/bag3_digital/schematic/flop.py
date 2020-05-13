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

# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__flop(Module):
    """Module for library bag3_digital cell flop.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'flop.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length.',
            w_p='PMOS width.',
            w_n='NMOS width.',
            th_p='PMOS threshold.',
            th_n='NMOS threshold.',
            seg_m='Master segments dictionary.',
            seg_s='Slave segments dictionary.',
            seg_ck='segments of clock inverter.  0 to disable.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg_ck=0,
        )

    def design(self, lch: int, w_p: int, w_n: int, th_p: str, th_n: str, seg_m: Dict[str, int],
               seg_s: Dict[str, int], seg_ck: int) -> None:
        self.instances['XM'].design(lch=lch, w_p=w_p, w_n=w_n, th_p=th_p, th_n=th_n,
                                    seg_dict=seg_m)
        self.instances['XS'].design(lch=lch, w_p=w_p, w_n=w_n, th_p=th_p, th_n=th_n,
                                    seg_dict=seg_s)
        if seg_ck == 0:
            self.add_pin('clkb', TermType.input)
            self.delete_instance('XB')
        else:
            self.instances['XB'].design(seg=seg_ck, lch=lch, w_p=w_p, w_n=w_n, th_p=th_p, th_n=th_n)

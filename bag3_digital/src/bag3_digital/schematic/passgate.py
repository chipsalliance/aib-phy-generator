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

from typing import Dict, Any, Union, Optional

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__passgate(Module):
    """Module for library bag3_digital cell passgate.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'passgate.yaml'))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            seg='segments of passgate',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            w_p='pmos width.',
            w_n='nmos width.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            out_cap_large='True if output parasitic cap is large.  Only affects behavioral model.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            out_cap_large=None,
        )

    def design(self, lch: Union[int, float], seg: int, seg_p: int, seg_n: int, w_p: int, w_n: int,
               th_p: str, th_n: str, out_cap_large: Optional[bool]) -> None:
        if seg_p < 0:
            seg_p = seg
        if seg_n < 0:
            seg_n = seg
        if seg_p < 0 or seg_n < 0:
            raise ValueError('Invalid number of segments.')

        self.design_transistor('XN', w_n, lch, seg_n, th_n, m='')
        self.design_transistor('XP', w_p, lch, seg_p, th_p, m='')

        self.set_pin_attribute('d', 'type', 'trireg')
        if out_cap_large is not None:
            self.set_pin_attribute('d', 'trireg_cap_large', str(out_cap_large))

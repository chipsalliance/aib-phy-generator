# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from typing import Dict, Any, Optional

import pkg_resources
from pathlib import Path

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__inv_tristate(Module):
    """Module for library bag3_digital cell inv_tristate.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'inv_tristate.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            seg='segments of transistors.',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            lch='channel length.',
            w_p='PMOS width.',
            w_n='NMOS width.',
            th_p='PMOS threshold.',
            th_n='NMOS threshold.',
            has_rsthb='True to add reset-high-bar pin.',
            out_cap_large='True if output parasitic cap is large.  Only affect behavioral model.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            has_rsthb=False,
            out_cap_large=None,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
        )

    def design(self, seg: int, seg_p: int, seg_n: int, lch: int, w_p: int, w_n: int, th_p: str,
               th_n: str, has_rsthb: bool, out_cap_large: Optional[bool], stack_p: int,
               stack_n: int) -> None:
        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Cannot have negative number of segments.')

        # in net for pmos and nmos with stacking
        pin_list = ['enb'] * stack_p + ['in'] * stack_p
        nin_list = ['en'] * stack_n + ['in'] * stack_n
        nin_name = ','.join(nin_list)
        pin_name = ','.join(pin_list)

        self.instances['XP'].design(w=w_p, lch=lch, seg=seg_p, intent=th_p, stack=2 * stack_p)
        pg_name = f'g<{2 * stack_p - 1}:0>'
        self.reconnect_instance_terminal('XP', pg_name, pin_name)
        if has_rsthb:
            self.add_pin('rsthb', TermType.input)
            self.instances['XN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=3)
            self.reconnect_instance_terminal('XN', 'g<2:0>', 'en,in,rsthb')
            self.instances['XR'].design(w=w_p, l=lch, nf=seg_p, intent=th_p)
            self.reconnect_instance_terminal('XR', 'G', 'rsthb')
            if stack_n > 1:
                # TODO
                raise ValueError(f'stack_n > 1 not supported if has_rsthb is True')
        else:
            self.remove_instance('XR')
            self.instances['XN'].design(w=w_n, lch=lch, seg=seg_n, intent=th_n, stack=2 * stack_n)
            ng_name = f'g<{2 * stack_n - 1}:0>'
            self.reconnect_instance_terminal('XN', ng_name, nin_name)

        self.set_pin_attribute('out', 'type', 'trireg')
        if out_cap_large is not None:
            self.set_pin_attribute('out', 'trireg_cap_large', str(out_cap_large))

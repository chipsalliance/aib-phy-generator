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

from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__flop_scan_rstlb(Module):
    """Module for library bag3_digital cell flop_scan_rstlb.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'flop_scan_rstlb.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            lch='channel length.',
            seg_dict='segments dictionary.',
            w_dict='width dictionary.',
            th_p='PMOS threshold.',
            th_n='NMOS threshold.',
        )

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int],
               th_p: str, th_n: str) -> None:
        p_buf = w_dict['p_buf']
        n_buf = w_dict['n_buf']
        p_in = w_dict['p_in']
        n_in = w_dict['n_in']
        p_mux = w_dict['p_mux']
        n_mux = w_dict['n_mux']
        p_keep = w_dict['p_keep']
        n_keep = w_dict['n_keep']
        p_pass = w_dict['p_pass']
        n_pass = w_dict['n_pass']
        p_rst = w_dict['p_rst']
        n_rst = w_dict['n_rst']
        p_out = w_dict['p_out']
        n_out = w_dict['n_out']

        seg_buf = seg_dict['buf']
        seg_in = seg_dict['in']
        seg_mux = seg_dict['mux']
        seg_keep = seg_dict['keep']
        seg_pass = seg_dict['pass']
        seg_rst = seg_dict['rst']
        seg_out = seg_dict['out']

        self.instances['XSUM0'].design(nin=2)
        self.instances['XSUM1'].design(nin=2)
        self.instances['XSUM2'].design(nin=2)
        self.instances['XCLK'].design(lch=lch, w_p=p_buf, w_n=n_buf, th_p=th_p, th_n=th_n,
                                      seg=seg_buf)
        self.instances['XSE'].design(lch=lch, w_p=p_buf, w_n=n_buf, th_p=th_p, th_n=th_n,
                                     seg=seg_buf)

        self.instances['XIN'].design(seg=seg_in, lch=lch, w_p=p_in, w_n=n_in, th_p=th_p, th_n=th_n)
        self.instances['XSI'].design(seg=seg_in, lch=lch, w_p=p_in, w_n=n_in, th_p=th_p, th_n=th_n)
        self.instances['XMUX'].design(seg=seg_mux, lch=lch, w_p=p_mux, w_n=n_mux,
                                      th_p=th_p, th_n=th_n)
        self.instances['XKEEP'].design(seg=seg_keep, lch=lch, w_p=p_keep, w_n=n_keep,
                                       th_p=th_p, th_n=th_n)
        self.instances['XNAND'].design(lch=lch, w_p=p_rst, w_n=n_rst,
                                       th_p=th_p, th_n=th_n, num_in=2, seg=seg_rst)
        self.instances['XPASS'].design(seg=seg_pass, lch=lch, w_p=p_pass, w_n=n_pass,
                                       th_p=th_p, th_n=th_n)
        self.instances['XRST'].design(seg=seg_rst, lch=lch, w_p=p_rst, w_n=n_rst,
                                      th_p=th_p, th_n=th_n, has_rsthb=True)
        self.reconnect_instance_terminal('XRST', 'rsthb', 'rstlb')

        self.instances['XFB'].design(lch=lch, w_p=p_out, w_n=n_out, th_p=th_p, th_n=th_n,
                                     seg=seg_out)
        self.instances['XOUT'].design(lch=lch, w_p=p_out, w_n=n_out, th_p=th_p, th_n=th_n,
                                      seg=seg_out)

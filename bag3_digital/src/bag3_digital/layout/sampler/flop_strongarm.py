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

"""This module contains a strong arm based flop"""

from typing import Any, Dict, Type, Optional

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.template import TemplateBase, TemplateDB

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ...schematic.flop_strongarm import bag3_digital__flop_strongarm
from .sr_latch import SRLatchSymmetric
from .strongarm import SAFrontend
from .strongarm_dig import SAFrontendDigital


class FlopStrongArm(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__flop_strongarm

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The placement information object.',
            sa_params='StrongArm parameters.',
            sr_params='SR latch parameters.',
            has_rstlb='True to enable rstlb functionality.',
            swap_outbuf='True to swap output buffers, so outp is on opposite side of inp.',
            out_pitch='output wire pitch from center.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            has_rstlb=False,
            swap_outbuf=False,
            out_pitch=0.5,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        sa_params: Param = self.params['sa_params']
        sr_params: Param = self.params['sr_params']
        has_rstlb: bool = self.params['has_rstlb']
        swap_outbuf: bool = self.params['swap_outbuf']
        out_pitch: HalfInt = HalfInt.convert(self.params['out_pitch'])

        # create masters
        sr_pinfo = self.get_tile_pinfo(1)

        sr_params = sr_params.copy(append=dict(pinfo=sr_pinfo, has_rstb=has_rstlb, has_inbuf=True,
                                               swap_outbuf=swap_outbuf, out_pitch=out_pitch))
        sr_master: SRLatchSymmetric = self.new_template(SRLatchSymmetric, params=sr_params)

        even_center = sr_master.num_cols % 4 == 0
        sa_pinfo = self.get_tile_pinfo(0)
        dig_style = (sa_pinfo == sr_pinfo)
        if dig_style:
            # digital style
            sa_params = sa_params.copy(append=dict(pinfo=sa_pinfo, has_rstb=has_rstlb,
                                                   even_center=even_center))
            sa_master: MOSBase = self.new_template(SAFrontendDigital, params=sa_params)
        else:
            # analog style
            sa_params = sa_params.copy(append=dict(pinfo=sa_pinfo, has_rstb=has_rstlb,
                                                   even_center=even_center, vertical_out=False,
                                                   vertical_rstb=False))
            sa_master: MOSBase = self.new_template(SAFrontend, params=sa_params)

        # placement
        sa_ncol = sa_master.num_cols
        sr_ncol = sr_master.num_cols
        ncol = max(sa_ncol, sr_ncol)

        sa = self.add_tile(sa_master, 0, (ncol - sa_ncol) // 2)
        # NOTE: flip SR latch so outputs and inputs lines up nicely
        sr = self.add_tile(sr_master, 1, (ncol - sr_ncol) // 2 + sr_ncol, flip_lr=True)
        self.set_mos_size()

        # supplies
        self.add_pin('VSS', [sr.get_pin('VSS'), sa.get_pin('VSS')], connect=True)
        vdd = self.connect_wires([sr.get_pin('VDD'), sa.get_pin('VDD')])[0]
        self.add_pin('VDD', vdd)

        # connect outputs
        rb = sr.get_pin('rb')
        sb = sr.get_pin('sb')
        outp = sa.get_all_port_pins('outp')
        outn = sa.get_all_port_pins('outn')
        self.connect_to_track_wires(outp, rb)
        self.connect_to_track_wires(outn, sb)

        # connect reset signals
        if has_rstlb:
            rstlb = sr.get_pin('rstlb')
            rsthb = sr.get_pin('rsthb')
            if dig_style:
                sa_rstb = sa.get_pin('rstb')
                rstlbl = self.connect_to_track_wires(sa_rstb, rstlb)
                rsthb = self.connect_to_tracks(vdd, rsthb.track_id, track_lower=rstlbl.lower,
                                               track_upper=rsthb.upper)
                self.add_pin('rstlb', sa_rstb)
                self.add_pin('rsthb', rsthb, hide=True)
            else:
                sa_rstbl = [sa.get_pin('prstbl'), sa.get_pin('nrstb')]
                sa_rstbr = [sa.get_pin('prstbr'), sa_rstbl[1]]
                rstlbl = self.connect_to_track_wires(sa_rstbl, rstlb)
                rstlbr = self.connect_to_tracks(sa_rstbr, rsthb.track_id)
                self.connect_to_tracks(rsthb, vdd.track_id)
                self.add_pin('rstlb', sa_rstbl[1])
                self.add_pin('rstlb_vm_r', rstlbr, hide=True)

            self.add_pin('rstlb_vm_l', rstlbl, hide=True)

        # reexport pins
        for name in ['inp', 'inn', 'clk', 'clkl', 'clkr']:
            self.reexport(sa.get_port(name))
        self.reexport(sr.get_port('q'), net_name='outp')
        self.reexport(sr.get_port('qb'), net_name='outn')

        self.sch_params = dict(
            sa_params=sa_master.sch_params.copy(remove=['has_rstb']),
            sr_params=sr_master.sch_params.copy(remove=['has_rstb']),
            has_rstlb=has_rstlb
        )

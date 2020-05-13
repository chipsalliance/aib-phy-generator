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

"""This package contains design class for PhaseDetector"""

from typing import Dict, Any, Tuple, List, Mapping, Sequence

import pprint
from pathlib import Path

from bag.simulation.cache import DesignInstance
from bag.design.netlist import add_mismatch_offsets
from bag.env import get_tech_global_info

from xbase.layout.enum import MOSType
from xbase.layout.mos.base import MOSBasePlaceInfo
from xbase.layout.mos.top import GenericWrapper

from bag3_digital.design.base import DigitalDesigner
from bag3_digital.design.stdcells.inv.cin_match import InvCapInMatchDesigner
from bag3_digital.layout.sampler.flop_strongarm import FlopStrongArm
from bag3_digital.measurement.util import get_in_buffer_pin_names
from bag3_digital.measurement.cap.delay_match import CapDelayMatch

from aib_ams.design.se_to_diff_en import SingleToDiffDesigner
from aib_ams.layout.phase_det import PhaseDetector
from aib_ams.measurement.phase_det import PhaseDetMeasManager


class PhaseDetectorDesigner(DigitalDesigner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._flop_params: Mapping[str, Any] = {}
        self._td_cin_flop_specs: Dict[str, Any] = {}
        self._inv_dsn_specs: Dict[str, Any] = {}
        self._flop_clk_pin: str = ''
        self._se_to_diff_dsn_params: Dict[str, Any] = {}
        self._phase_det_mm_params = dict()
        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.dsn_specs
        tile_specs: Mapping[str, Any] = specs['tile_specs']
        dig_tran_specs: Mapping[str, Any] = specs['dig_tran_specs']
        sup_values: Mapping[str, Any] = specs['sup_values']
        c_load: float = specs['c_load']
        tiles_flop: Sequence[Mapping[str, Any]] = specs['tiles_flop']
        flop_params: Mapping[str, Any] = specs['flop_params']
        cin_params: Mapping[str, Any] = specs['cin_params']
        w_min: int = specs['w_min']
        w_res: int = specs['w_res']
        sign_off_envs: Sequence[str] = specs['sign_off_envs']
        se2diff_params: Mapping[str, Any] = specs['se2diff_params']
        t_rf: float = specs['t_rf']

        buf_params: Mapping[str, Any] = cin_params['buf_params']
        search_params: Mapping[str, Any] = cin_params['search_params']
        err_targ: float = cin_params['err_targ']
        search_step: int = cin_params['search_step']

        self._flop_params = dict(pinfo=self.make_tile_pattern(tiles_flop), **flop_params)

        pwr_tup = ('VSS', 'VDD')
        start_pin, stop_pin = get_in_buffer_pin_names('clk')
        pwr_domain = {p_: pwr_tup for p_ in ['inp', 'inn', 'outp', 'outn', 'clk', 'rstlb']}
        supply_map = dict(VDD='VDD', VSS='VSS')
        pin_values = dict(inp=1, inn=0)
        reset_list = [('rstlb', False)]
        diff_list = [(['inp'], ['inn']), (['outp'], ['outn'])]
        flop_tbm_specs = self.get_dig_tran_specs(pwr_domain, supply_map, pin_values=pin_values,
                                                 reset_list=reset_list, diff_list=diff_list)

        flop_tbm_specs['sim_params'] = sim_params = dict(**flop_tbm_specs['sim_params'])
        sim_params['c_out'] = c_load
        self._td_cin_flop_specs = dict(
            in_pin='clk',
            buf_params=buf_params,
            search_params=search_params,
            tbm_specs=flop_tbm_specs,
            load_list=[dict(pin='outp', type='cap', value='c_out'),
                       dict(pin='outn', type='cap', value='c_out')],
        )

        self._flop_clk_pin = stop_pin

        sa_tile = self.get_tile('strongarm')
        ridx_n = 0
        ridx_p = -1
        w_n = sa_tile.get_row_place_info(ridx_n).row_info.width
        w_p = sa_tile.get_row_place_info(ridx_p).row_info.width
        seg_dict = flop_params['sa_params']['seg_dict']
        seg = max(seg_dict['tail'], 4 * seg_dict['sw'])
        self._inv_dsn_specs = dict(
            dig_tran_specs=dig_tran_specs,
            sup_values=sup_values,
            buf_params=buf_params,
            ridx_n=ridx_n,
            ridx_p=ridx_p,
            w_min=w_min,
            w_res=w_res,
            c_load=0,
            err_targ=err_targ,
            search_step=search_step,
            inv_params_init=dict(w_n=w_n, w_p=w_p, seg=seg),
            tile_name='strongarm',
            tile_specs=tile_specs,
        )

        self._se_to_diff_dsn_params = dict(
            dig_tran_specs=dig_tran_specs,
            sup_values=sup_values,
            t_rf=t_rf,
            w_min=w_min,
            w_res=w_res,
            tile_name='logic',
            tile_specs=tile_specs,
            sign_off_envs=sign_off_envs,
            **se2diff_params,
        )

        self._phase_det_mm_params = dict(
            out_rising=False,
            accept_mode=False,
            sigma_avt=0.001,
            sim_envs=sign_off_envs,
            **specs['meas_specs'],
        )
        self._phase_det_mm_params['tbm_specs']['sim_envs'] = sign_off_envs

    async def async_design(self, dig_tran_specs: Mapping[str, Any], sup_values: Mapping[str, Any], c_load: float,
                           w_min: int, w_res: int, t_rf: float, se2diff_params: Mapping[str, Any],
                           flop_params: Mapping[str, Any], cin_params: Mapping[str, Any], meas_specs: Mapping[str, Any],
                           sign_off_envs: Sequence[str], vm_pitch: float, tile_se_to_diff: str, tile_dummy: str,
                           tiles_flop: Sequence[Mapping[str, Any]], tiles_phasedet: Sequence[Mapping[str, Any]],
                           tile_specs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Passed in kwargs are the same as self.dsn_specs

        Parameters
        ----------
        dig_tran_specs : Mapping[str, Any]
            DigitalTranTB testbench specs
        sup_values : Mapping[str, Any]
            The Supply Values
        c_load : Mapping[str, Any]
            Load Cap for this block
        w_min : int
            The minimum width
        w_res : int
            The width resolution
        t_rf : float
            The Rise-Fall time of the clocks
        se2diff_params : Mapping[str, Any]
            Design Parameters for the SE to Diff used inside this block
        flop_params : Mapping[str, Any]
            Layout Parameters for the Strong Arm Flops in the design
        cin_params : Mapping[str, Any]
            Cin measurement params
        meas_specs : Mapping[str, Any]
            Specs for the phase detector measurement manager
        sign_off_envs : Sequence[str]
            The environments used for signoff
        vm_pitch : float
            The pitch of the vm layer
        tile_se_to_diff : str
            The name of the tile that the SE to Diff is on
        tile_dummy : str
            The name of the tile to place dummies on
        tiles_flop : Sequence[Mapping[str, Any]]
            A dictionary for each of the flop tiles, containing the key "name" for the name of the tile,
            and the key flip if the tile should be flipped
        tiles_phasedet : Sequence[Mapping[str, Any]]
            A List of dictionaries, containing the key "name" for the name of the tile, and a the key "flip" with the
            value specifying if the tile should be flipped. `tiles_phasedet` should contain the order for all the tiles
            used in the phase_det
        tile_specs : Mapping[str, Any]
            A dictionary of tile specification data
        Returns
        -------
        ans : Mapping[str, Any]
            Design summary
        """
        # Start by extracting the cap on
        td_rise, c_clk_rise, c_in = await self.get_flop_td_cin()
        # Design dummy inverter
        inv_params = await self.get_inv_params(td_rise)
        # Design the Single-ended to Diff converter
        self._se_to_diff_dsn_params['c_load'] = c_clk_rise + c_in
        dsn = self.new_designer(SingleToDiffDesigner, self._se_to_diff_dsn_params)
        se2diff_summary = await dsn.async_design(**self._se_to_diff_dsn_params)
        se2diff_params = se2diff_summary['se_params']
        se2diff_params.pop('pinfo')
        se2diff_params['export_pins'] = False
        # Characterize the se2diff and the strong arm to determine the td_off
        dut_params = dict(
            cls_name=PhaseDetector.get_qualified_name(),
            params=dict(
                pinfo=self.make_tile_pattern(tiles_phasedet),
                vm_pitch=vm_pitch,
                se_params=se2diff_params,
                flop_params=flop_params,
                inv_params=inv_params
            )
        )
        results = await self.verify_design(dut_params)
        return dict(inv_params=inv_params, c_flop=c_in + c_clk_rise, results=results)

    async def get_inv_params(self, clk_rise: float) -> Tuple[Mapping[str, Any], float, float]:
        """
        Parameters
        ----------
        clk_rise : float
            The rise time of the clock input in the Strong arm flop

        Returns
        -------
        inv_params : Mapping[str, Any]
            Parameters for the Dummy Inverter
        """
        # Note: care only about rising edge on flop clk = falling edge on inverter input
        self._inv_dsn_specs['td_fall'] = clk_rise
        inv_dsn = self.new_designer(InvCapInMatchDesigner, self._inv_dsn_specs)
        result = await inv_dsn.async_design()
        inv_params = result['inv_params']
        self.log(f'inv_params:\n{pprint.pformat(inv_params, width=100)}')
        return inv_params

    async def get_flop_td_cin(self) -> Tuple[float, float, float]:
        """

        Returns
        -------
        clk_td_rise : float
            The clk rise time
        c_clk_rise : float
            The capacitance on the clock input of the flop with clock rising
        c_in_avg : float
            Average capacitance on the inputs of the clock
        """
        dut_flop = await self.async_wrapper_dut('FLOP_SA', FlopStrongArm, self._flop_params)

        mm = self.make_mm(CapDelayMatch, self._td_cin_flop_specs)
        mm_result = await self.async_simulate_mm_obj('c_clk_flop', dut_flop, mm)
        data = mm_result.data
        clk_td_fall = data['tf_ref']
        clk_td_rise = data['tr_ref']
        c_clk_rise = data['cap_rise']

        mm.specs['in_pin'] = 'inp'
        mm.commit()
        mm_result = await self.async_simulate_mm_obj('c_in_flop', dut_flop, mm)
        data = mm_result.data
        c_in_rise = data['cap_rise']
        c_in_fall = data['cap_fall']
        c_in_avg = (c_in_fall + c_in_rise) / 2
        self.log(f'flop cap results:\nclk_td_fall={clk_td_fall:.4g}, '
                 f'clk_td_rise={clk_td_rise:.4g}\n'
                 f'c_clk_rise={c_clk_rise:.4g}, c_in_fall={c_in_fall:.4g}, '
                 f'c_in_rise={c_in_rise:.4g}, c_in_avg={c_in_avg}')
        return clk_td_rise, c_clk_rise, c_in_avg

    async def verify_design(self, dut_params: Mapping[str, Any]) -> Mapping[str, Any]:
        dut = await self.async_new_dut('phase_det', GenericWrapper, dut_params, flat=True)
        offset_netlist = Path(*dut.netlist_path.parts[:-1], 'netlist_with_offsets.scs')
        v_offset_map = add_mismatch_offsets(dut.netlist_path, offset_netlist, self._sim_db._sim.netlist_type)
        designed_dut_with_offsets = DesignInstance(dut.cell_name, dut.sch_master, dut.lay_master,
                                                   offset_netlist, dut.cv_info_list)
        seg_dict: Mapping[str, Any] = dut_params['params']['flop_params']['sa_params']['seg_dict']
        global_params = get_tech_global_info("aib_ams")
        a_vt_per_fin = global_params['A_vt_fin_n']
        seg_in = seg_dict['in']
        seg_tail = seg_dict['tail']
        offset_tail = 3*(a_vt_per_fin/seg_tail)**(1/2)
        offset_in = 3*(a_vt_per_fin/seg_in)**(1/2)
        strong_arm_offsets = dict(
            XFLOPD_XSA_XTAIL=-offset_tail,
            XFLOPU_XSA_XTAIL=-offset_tail,
            XFLOPD_XSA_XINP=offset_in,
            XFLOPD_XSA_XINN=-offset_in,
            XFLOPU_XSA_XINP=offset_in,
            XFLOPU_XSA_XINN=-offset_in,
        )
        for mos_name, voff_name in v_offset_map.items():
            found = False
            for base_name, offset in strong_arm_offsets.items():
                if mos_name.startswith(base_name):
                    self._phase_det_mm_params['tbm_specs']['sim_params'][voff_name] = offset
                    found = True
            if not found:
                self._phase_det_mm_params['tbm_specs']['sim_params'][voff_name] = 0
        mm = self.make_mm(PhaseDetMeasManager, self._phase_det_mm_params)
        results = await self.async_simulate_mm_obj(f'phase_det_timing_with_offset', designed_dut_with_offsets, mm)
        return results.data

    @staticmethod
    def _get_default_width(pinfo: MOSBasePlaceInfo) -> Tuple[int, int]:
        wn, wp = [], []
        for row_place_info in map(pinfo.get_row_place_info, range(pinfo.num_rows)):
            w = row_place_info.row_info.width
            if row_place_info.row_info.row_type is MOSType.nch:
                wn.append(w)
            elif row_place_info.row_info.row_type is MOSType.pch:
                wp.append(w)
        # In the case that there are multiple NMOS or PMOS rows, this function returns the
        # most strict constraint. Typically, the width ends up being the same anyway.
        if len(wn) > 1:
            wn = [min(wn)]
        if len(wp) > 1:
            wp = [min(wp)]
        return wn[0], wp[0]

    @classmethod
    def _get_cap_delay_match_mm_params(cls, cap_mm_params: Mapping[str, Any], vdd: float,
                                       freq: float, trf_in: float, row_info: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        pins = ['in', 'out']
        pwr_domain = {pin: ('VSS', 'VDD') for pin in pins}
        tbm_specs = cap_mm_params['tbm_specs']
        tbm_specs['sim_params'].update(dict(
            t_rst=5 * trf_in,
            t_rst_rf=trf_in,
            t_bit=1.0 / freq,
            t_rf=trf_in,
        ))
        tbm_specs['pwr_domain'] = pwr_domain
        tbm_specs['sup_values'] = dict(VDD=vdd, VSS=0.0)

        try:
            inv_params_list = cap_mm_params['buf_params']['inv_params']
            assert len(inv_params_list) == 2
        except KeyError:
            inv_params_list = [{}, {}]

        load_list = [dict(pin='out', type='cap', value='c_out')]

        return dict(
            tbm_specs=tbm_specs,
            buf_params=cls._get_buf_params(inv_params_list, row_info),
            search_params=cap_mm_params['search_params'],
            load_list=load_list
        )

    @classmethod
    def _get_phase_det_mm_params(cls, phase_det_mm_params: Mapping[str, Any],
                                 tbm_specs: Mapping[str, Any],
                                 strongarm_offset_params: Mapping[str, Any],
                                 res_timing_err: float, sigma_avt: float,
                                 row_info: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            clk_pins=['CLKA', 'CLKB'],
            out_pins=['t_up', 't_down'],
            res_timing_err=res_timing_err,
            tbm_specs=tbm_specs,
            buf_params=cls._get_buf_params([{'seg': 8}, {'seg': 8}], row_info),
            strongarm_offset_params=strongarm_offset_params,
            sigma_avt=sigma_avt,
            **phase_det_mm_params
        )

    @staticmethod
    def _get_buf_params(inv_params_list: List[Dict[str, Any]], row_info: Dict[str, Any]):
        for inv_params in inv_params_list:
            for var in ['lch', 'w_p', 'w_n', 'th_n', 'th_p']:
                if var not in inv_params:
                    inv_params[var] = row_info[var]

        return dict(
            inv_params=inv_params_list,
            export_pins=True
        )

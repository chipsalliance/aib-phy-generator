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

from typing import Mapping, Dict, Any, Optional, cast, Type

import numpy as np

from bag.layout.template import TemplateBase
from bag.simulation.design import DesignerBase
from bag.concurrent.util import GatherHelper
from bag.env import get_tech_global_info

from xbase.layout.mos.placement.data import TileInfoTable

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType

from bag3_digital.design.lvl_shift import LvlShiftDesigner
from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag3_digital.measurement.cap.delay_match import CapDelayMatch

from .output_driver import OutputDriverDesigner
from ..layout.txanlg import TXAnalog


class TXAnalogCoreDesigner(DesignerBase):
    """ Design the TX Analog Cell """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self.dsn_tech_info = get_tech_global_info('aib_ams')

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return TXAnalog

    async def async_design(self,
                           num_units: int,
                           num_units_nom: int,
                           num_units_min: int,
                           r_targ: float,
                           r_min_weak: float,
                           c_ext: float,
                           freq: float,
                           trf_max: float,
                           trf_in: float,
                           trst: float,
                           k_ratio_ctrl: float,
                           k_ratio_data: float,
                           rel_err: float,
                           del_err: float,
                           td_max: float,
                           stack_max: int,
                           tile_name: str,
                           tile_specs: Mapping[str, Any],
                           dig_tbm_specs: Dict[str, Any],
                           dig_buf_params: Dict[str, Any],
                           cap_in_search_params: Dict[str, Any],
                           res_mm_specs: Dict[str, Any],
                           ridx_p: int = 1,
                           ridx_n: int = 1,
                           tran_options: Optional[Mapping[str, Any]] = None,
                           inv_input_cap_meas_seg: Optional[int] = 16,
                           sim_options: Optional[Mapping[str, Any]] = None,
                           mc_params: Optional[Dict[str, Any]] = None,
                           mc_corner: Optional[str] = '',
                           specs_yaml_fname: Optional[str] = 'txanlg_specs.yaml'
                           ) -> Mapping[str, Any]:

        """This function runs top level design on the TX.
        1) Call OutputDriverDesigner to design driver
        2) Use CapDelayMatch to determine output driver input capacitance
        3) Design data and control level shifters
        4) Sign off

        Parameters
        ----------
        num_units: int
            Max. number of units.
        num_units_nom: int
            Number of units used in nominal case.
        num_units_min:
            Min. number of units used.
        r_targ: float
            Target nominal (strong) output resistance.
        r_min_weak: float
            Target (weak) output resistance.
        c_ext: float
            Nominal Loading Cap.
        freq: float
            The data frequency.
        trf_max: float
            Max. rise/fall time requirement.
        trf_in: float
            Rise/fall time at input.
        trst: float
            Reset signal arrival time compared to data edge.
        k_ratio_ctrl: float
            The strength ratio of nmos to pmos in control signal level shifters.
        k_ratio_data: float
            The strength ratio of nmos to pmos in data signal level shifters.
        rel_err: float
            Output resistance error tolerance, used in DriverPullUpDownDesigner
        del_err: float
            Delay mismatch tolerance, used in DriverUnitCellDesigner for sizing NAND + NOR
        td_max: float
            Max. delay on data path.
        stack_max: int
            Max. number of stacks possible for pull up / pull down driver
        tile_name: str
            Tile name for layout.
        tile_specs: Mapping[str, Any]
            Tile specifications for layout.
        dig_tbm_specs: Mapping[str, Any]
            DigitalTranTB params
        dig_buf_params: Mapping[str, Any]
            Digital buffer params
        cap_in_search_params: Mapping[str, Any]
            Search parameters for finding cin of blocks
        res_mm_specs: Mapping[str, Any]
            Specs for DriverPullUpDownMM, used in DriverPullUpDownDesigner
        ridx_n: int
            NMOS transistor row
        ridx_p: int
            PMOS transistor row
        inv_input_cap_meas_seg: Optional[int]
            When determining inverter input capacitance, use this size inverter as reference
        tran_options: Optional[Mapping[str, Any]]
            Additional transient simulation options dictionary, used in DriverPullUpDownDesigner
        sim_options: Optional[Mapping[str, Any]]
            Simulator-specific simulation options
        mc_params: Optional[Dict[str, Any]]
            Monte Carlo simulation parameters
        mc_corner: Optional[str]
            Corner to run Monte Carlo simulations
        specs_yaml_fname:
            Output file location to write designed generator parameters

        Returns
        -------
        summary: Mapping[str, Any]
            Design summary

        """

        # Get and set max widths of final drivers from tech defaults; set this in tile info
        tech_info = self.dsn_tech_info
        w_p = tech_info['w_maxp']
        w_n = tech_info['w_maxn']
        if 'lch' not in tile_specs['arr_info']:
            tile_specs['arr_info']['lch'] = tech_info['lch_min']
        tile_specs['place_info'][tile_name]['row_specs'][0]['width'] = w_n
        tile_specs['place_info'][tile_name]['row_specs'][1]['width'] = w_p
        tinfo_table = TileInfoTable.make_tiles(self.grid, tile_specs)
        pinfo = tinfo_table[tile_name]

        # Design the TX Output Driver and get its nominal delay
        driver_design_specs = dict(
            num_units=num_units, num_units_nom=num_units_nom, num_units_min=num_units_min,
            r_targ=r_targ, r_min_weak=r_min_weak, c_ext=c_ext, trf_max=trf_max,
            stack_max=stack_max, freq=freq, trf_in=trf_in, rel_err=rel_err,
            del_err=del_err, tile_name=tile_name, tile_specs=tile_specs,
            tran_options=tran_options, res_mm_specs=res_mm_specs)
        driver_designer = OutputDriverDesigner(self._root_dir, self._sim_db, self._dsn_specs)
        driver_designer.set_dsn_specs(driver_design_specs)
        driver_summary = await driver_designer.async_design(**driver_design_specs)
        print('-------')
        print('Driver design complete.')
        print(driver_summary)

        td_driver = np.max([driver_summary['tdr'], driver_summary['tdf']])
        td_lv_shift = td_max - np.max(td_driver)
        print('Required level shift delay is: ', td_lv_shift)
        if td_lv_shift < 0:
            raise ValueError("Output driver delay exceeds maximum allowed for entire circuit.")

        # Calculate the input cap of a 1 segment inverter (Required for lvlshifters)
        # sim_env_params = tech_info['dsn_envs']['center']
        sim_env_params = tech_info['dsn_envs']['slow_io']
        vdd_out = sim_env_params['vddio']
        dig_tbm_specs['sim_envs'] = sim_env_params['env']
        inv_input_cap = await self._get_inv_input_cap(pinfo, vdd_out,
                                                      inv_input_cap_meas_seg,
                                                      dig_tbm_specs, dig_buf_params,
                                                      cap_in_search_params)

        inv_input_cap_per_seg = inv_input_cap / inv_input_cap_meas_seg
        print(f'Inverter input capacitance per segment is {inv_input_cap_per_seg}')
        input_cap_per_fin = inv_input_cap_per_seg / (dig_buf_params['inv_params'][0]['w_p'] +
                                                     dig_buf_params['inv_params'][0]['w_n'])

        # Get input cap of all driver pins
        driver_cdict = await self.get_driver_cin(driver_summary, vdd_out, c_ext,
                                                 cap_in_search_params,
                                                 dig_buf_params, dig_tbm_specs, num_units)
        print(f'Driver input capacitance is {driver_cdict["data"]}')

        # Design level shifters
        lv_params = dict(trf_in=trf_in, tile_specs=tile_specs, k_ratio_ctrl=k_ratio_ctrl,
                         k_ratio_data=k_ratio_data, tile_name=tile_name,
                         inv_input_cap_per_seg=inv_input_cap_per_seg,
                         inv_input_cap_per_fin=input_cap_per_fin, has_rst=True)
        lvshift_summary = {}
        for key, cload in driver_cdict.items():
            # td_lv_shift will be ignored in ctrl level shifters and max_fanout in tech_info will
            # be used as a design objective
            summ = await self._iter_dsn_lvshift(cload, td_lv_shift, is_ctrl=(key == 'ctrl'),
                                                dual_output=(key == 'ctrl'), **lv_params)
            lvshift_summary[key] = summ

        summary = dict(
            driver=driver_summary,
            data_lvshift=lvshift_summary['data'],
            ctrl_lvshift=lvshift_summary['ctrl']
        )

        lay_params = self.get_layout_params(summary, pinfo)
        dut = await self.async_new_dut('aib_txanlg', STDCellWrapper, lay_params)
        print("=" * 80)
        print("TX: Running signoff...")
        await self.signoff_dut(dut, c_ext, freq, trf_in, trst, td_max, trf_max,
                               mc_params=mc_params, mc_corner=mc_corner)

        self.write_out_specs_yaml(summary, specs_yaml_fname)

        return summary

    def write_out_specs_yaml(self, summary_in: Dict[str, Any], yaml_fname: str) -> None:
        from bag.io.file import write_yaml

        dict_to_write = summary_in.copy()
        self.prep_lvshift_dict(dict_to_write, 'ctrl_lvshift')
        self.prep_lvshift_dict(dict_to_write, 'data_lvshift')
        self.prep_driver_dict(dict_to_write, 'driver')

        write_yaml(yaml_fname, dict_to_write)
        self.log(f'Wrote final layout specifications to {yaml_fname}.')

    @staticmethod
    def prep_driver_dict(dict_to_write: Dict[str, Any], drvname: str) -> None:
        TXAnalogCoreDesigner.prep_dicts_common_params(dict_to_write, drvname)
        dict_to_write[drvname].pop('duty_err')
        dict_to_write[drvname].pop('tdf')
        dict_to_write[drvname].pop('tdr')
        dict_to_write[drvname].pop('trf_worst')

    @staticmethod
    def prep_lvshift_dict(dict_to_write: Dict[str, Any], lvname: str) -> None:
        TXAnalogCoreDesigner.prep_dicts_common_params(dict_to_write, lvname)
        dict_to_write[lvname]['dut_params'].pop('pwr_gnd_list')
        dict_to_write[lvname].pop('tdf')
        dict_to_write[lvname].pop('tdr')
        dict_to_write[lvname].pop('tint')
        dict_to_write[lvname].pop('worst_var')

    @staticmethod
    def prep_dicts_common_params(dict_to_write, subblock_name: str) -> None:
        dict_to_write[subblock_name]['dut_params'].pop('draw_taps')
        dict_to_write[subblock_name]['dut_params']['params'].pop('pinfo')

    @staticmethod
    def get_layout_params(summary_dict, pinfo):
        driver_params = summary_dict['driver']['dut_params']['params']
        data_lvshift_params = summary_dict['data_lvshift']['dut_params']['params']
        ctrl_lvshift_params = summary_dict['ctrl_lvshift']['dut_params']['params']
        lay_params = dict(
            cls_name=TXAnalog.get_qualified_name(),
            draw_taps=False,
            params=dict(
                pinfo=pinfo,
                drv_params=dict(
                    pupd_params=driver_params['pupd_params'],
                    unit_params=driver_params['unit_params'],
                ),
                data_lv_params=data_lvshift_params['lv_params'],
                ctrl_lv_params=ctrl_lvshift_params['lv_params'],
                buf_ctrl_lv_params=ctrl_lvshift_params['in_buf_params'],
                buf_data_lv_params=data_lvshift_params['in_buf_params'],
            )
        )
        return lay_params

    @staticmethod
    def get_tbm_specs(c_ext, freq, trf, trst, vdd_core, vdd_io, sim_envs, sim_options=None):
        tbm_specs = dict(
            dut_pins=['din', 'indrv_buf<1:0>', 'ipdrv_buf<1:0>', 'itx_en_buf', 'por_vccl',
                      'porb_vccl', 'weak_pulldownen', 'weak_pullupenb', 'txpadout', 'VDDCore',
                      'VDDIO', 'VSS'],
            pulse_list=[dict(pin='din',
                             tper='tbit',
                             tpw='tbit/2',
                             trf='trf',
                             pos=True
                             ),
                        ],
            load_list=[dict(pin='txpadout',
                            type='cap',
                            value=c_ext)],
            sup_values=dict(VSS=0,
                            VDDCore=vdd_core,
                            VDDIO=vdd_io,
                            ),
            pwr_domain={'din': ('VSS', 'VDDCore'),
                        'txpadout': ('VSS', 'VDDIO'),
                        'indrv_buf': ('VSS', 'VDDCore'),
                        'ipdrv_buf': ('VSS', 'VDDCore'),
                        'itx_en_buf': ('VSS', 'VDDCore'),
                        'por_vccl': ('VSS', 'VDDIO'),
                        'porb_vccl': ('VSS', 'VDDIO'),
                        'weak_pulldownen': ('VSS', 'VDDCore'),
                        'weak_pullupenb': ('VSS', 'VDDCore'),
                        },
            pin_values={'indrv_buf<1:0>': 3,
                        'ipdrv_buf<1:0>': 3,
                        'itx_en_buf': 1,
                        'weak_pulldownen': 0,
                        'weak_pullupenb': 1,
                        },
            reset_list=[('por_vccl', True), ('porb_vccl', False)],
            sim_envs=sim_envs,
            sim_params=dict(
                freq=freq,
                tbit=1/freq,
                trf=trf,
                t_rst=trst,
                t_rst_rf=trf,
                t_sim=2/freq,
            ),
            save_outputs=['din', 'txpadout']
        )
        if sim_options:
            tbm_specs['sim_options'] = sim_options

        return tbm_specs

    async def signoff_dut(self, dut, c_ext, freq, trf_in, trst, td_max, trf_max, sim_options=None,
                          run_lvl_extreme: bool = False, rel_err: float = 10e-2,
                          mc_params: dict = None, mc_corner: str = '') -> None:
        tech_info = self.dsn_tech_info
        all_corners = tech_info['signoff_envs']['all_corners']

        # Run level shifter extreme signoff
        if run_lvl_extreme:
            env = tech_info['signoff_envs']['lvl_func']['env']
            vdd_io = tech_info['signoff_envs']['lvl_func']['vddo']
            vdd_core = tech_info['signoff_envs']['lvl_func']['vddi']
            tbm_specs = self.get_tbm_specs(c_ext, freq, trf_in, trst, vdd_core, vdd_io, env,
                                           sim_options)
            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'signoff_txanlg_{env}', dut,
                                                            tbm, tbm_specs)
            td_in_hh = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                      in_edge=EdgeType.RISE, out_edge=EdgeType.RISE, t_start=1/freq)
            td_in_ll = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                      in_edge=EdgeType.FALL, out_edge=EdgeType.FALL, t_start=1/freq)
            td = max(td_in_hh, td_in_ll)
            if td < float('inf'):
                self.log('Design passed level shifter extreme signoff.')
            else:
                raise ValueError('Level shifter functionality signoff failed.')

        # Run all corners
        envs = all_corners['envs']
        worst_td = -float('inf')
        worst_duty_err = 0
        trf_worst = -float('inf')
        worst_env = ''
        trf_worst_env = ''
        worst_duty_env = ''
        sim_worst = None
        tbm_worst = None
        tbm = None
        for env in envs:
            vdd_io = all_corners['vddio'][env]
            vdd_core = all_corners['vdd'][env]
            tbm_specs = self.get_tbm_specs(c_ext, freq, trf_in, trst, vdd_core, vdd_io, [env],
                                           sim_options)

            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'signoff_txanlg_{env}', dut,
                                                            tbm, tbm_specs)

            td_in_hh: np.ndarray = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                                  in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                                  t_start=1/freq)
            td_in_ll: np.ndarray = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                                  in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                                  t_start=1/freq)
            td = max(td_in_hh, td_in_ll)
            if td[0] > worst_td:
                worst_td = td[0]
                worst_env = env
                sim_worst = sim_results
                tbm_worst = tbm

            td_duty_err = td_in_hh[0] - td_in_ll[0]
            if np.abs(td_duty_err) > np.abs(worst_duty_err):
                worst_duty_err = td_duty_err
                worst_duty_env = env

            trf_r = tbm_worst.calc_trf(sim_worst.data, 'txpadout', True)
            trf_f = tbm_worst.calc_trf(sim_worst.data, 'txpadout', False)
            trf = max(trf_r[0], trf_f[0])
            if trf > trf_worst:
                trf_worst = trf
                trf_worst_env = env

        msg = f'td_target = {td_max}, td_worst= {worst_td}, worst_env = {worst_env}'
        if worst_td > td_max:
            self.warn(msg)
        else:
            self.log(msg)
        self.log(f'Worst duty cycle error = {worst_duty_err} in corner {worst_duty_env}')
        self.log(f'Worst rise/fall time = {trf_worst}')
        if worst_td > td_max*(1+rel_err):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(sim_worst.data['time'].flatten(), sim_worst.data['din'].flatten(), 'b')
            plt.plot(sim_worst.data['time'].flatten(), sim_worst.data['txpadout'].flatten(), 'r')
            plt.title(f'Corner: {worst_env}')
            plt.show(block=False)
            raise RuntimeError(f'TX total delay exceeded target by more than {rel_err*100:.2f}%.')

        if trf_worst > trf_max*(1+rel_err):
            msg = f'TX rise/fall time exceeded specification by more than {rel_err*100:.2f}.\n' + \
                  f'Worst rise/fall corner: {trf_worst_env}'
            raise RuntimeError(msg)

        if mc_params and mc_corner:
            vdd_io = tech_info['dsn_envs'][mc_corner]['vddio']
            vdd_core = tech_info['dsn_envs'][mc_corner]['vdd']
            sim_envs = tech_info['dsn_envs'][mc_corner]['env']
            mc_tbm_specs = self.get_tbm_specs(c_ext, freq, trf_in, trst, vdd_core, vdd_io, sim_envs,
                                              sim_options)
            mc_tbm_specs['monte_carlo_params'] = mc_params
            mc_tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, mc_tbm_specs))
            sim_results = await self.async_simulate_tbm_obj(f'signoff_txanlg_monte', dut,
                                                            mc_tbm, mc_tbm_specs)

            td_in_hh = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                      in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                      t_start=1 / freq)
            td_in_ll = tbm.calc_delay(sim_results.data, 'din', 'txpadout',
                                      in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                      t_start=1 / freq)

            sigma_td_hh = np.sqrt(np.var(td_in_hh))
            sigma_td_ll = np.sqrt(np.var(td_in_ll))
            sigma_rf_mismatch = np.sqrt(np.var(td_in_hh - td_in_ll))

            self.log(f'Sigma rise: {sigma_td_hh}, Sigma fall: {sigma_td_ll}, ' +
                     f'Sigma r/f mismatch: {sigma_rf_mismatch}')

    async def _iter_dsn_lvshift(self, cload, d_targ, trf_in, tile_specs, k_ratio_data,
                                k_ratio_ctrl, tile_name,
                                inv_input_cap_per_seg, inv_input_cap_per_fin, is_ctrl: bool = False,
                                dual_output=False, has_rst=False) -> Dict[str, Any]:
        # Designs level shifter, if is_ctrl = True treats it as a ctrl lvshift
        fanout = 4 if not is_ctrl else self.dsn_tech_info['max_fanout']
        lv_shift_params = dict(
            cload=cload, dmax=d_targ, trf_in=trf_in, tile_specs=tile_specs,
            k_ratio=k_ratio_ctrl if is_ctrl else k_ratio_data, tile_name=tile_name,
            inv_input_cap=inv_input_cap_per_seg, inv_input_cap_per_fin=inv_input_cap_per_fin,
            fanout=fanout, dual_output=dual_output,
            has_rst=has_rst, vin='vdd', vout='vddio', is_ctrl=is_ctrl)
        lvshift_designer = LvlShiftDesigner(self._root_dir, self._sim_db, self._dsn_specs)
        lvshift_designer.set_dsn_specs(lv_shift_params)
        summary = await lvshift_designer.async_design(**lv_shift_params, exception_on_dmax=False)
        summary = cast(Dict[str, Any], summary)
        if is_ctrl:
            return summary
        td = max(max(summary['tdr']), max(summary['tdf']))
        self.log(f'Level shifter delay with fanout of {fanout} was {td}, wanted {d_targ}.')

        lvl_del_dict = dict(
            fanout=[],
            delay=[],
            tint=[],
        )

        while td > d_targ:
            tint = summary['tint']
            lvl_del_dict['fanout'].append(fanout)
            lvl_del_dict['delay'].append(td)
            lvl_del_dict['tint'].append(tint)
            self.log(f'Iterating on fanout of level shifter.')

            slope = (td - tint)/fanout
            fanout = 0.95*(d_targ - tint)/slope

            if fanout < 0:
                msg = f'ERROR: Level shifter\'s intrinsic delay is greater than the required' \
                      f' delay. d_targ = {d_targ}, t_int = {tint}'
                self.error(msg)

            if fanout <= 1:
                msg = 'Level shifter delay is too low and can not be achieved with fanout > 1.'
                self.error(msg)

            self.log(f'New fanout is {fanout}.')

            lv_shift_params['fanout'] = fanout
            lvshift_designer.set_dsn_specs(lv_shift_params)
            summary = await lvshift_designer.async_design(**lv_shift_params,
                                                          exception_on_dmax=False)
            td = max(max(summary['tdr']), max(summary['tdf']))
            self.log(f'Level shifter delay with fanout of {fanout} was {td}, wanted {d_targ}.')

        return summary

    async def get_driver_cin(self, driver_summary, vdd_out, c_ext, cap_in_search_params,
                             dig_buf_params, dig_tbm_specs, num_units) -> dict:

        pin_names = ['din', 'tristate', 'tristateb', 'weak_pden', 'weak_puenb',
                     'n_enb_drv<0>', 'n_enb_drv<1>', 'p_en_drv<0>', 'p_en_drv<1>']

        helper = GatherHelper()
        dut = await self.async_new_dut('cin_driver_dut', STDCellWrapper,
                                       driver_summary['dut_params'])
        for pin in pin_names:
            helper.append(self._get_driver_input_cap(pin, driver_summary['dut_params'], vdd_out,
                                                     c_ext, num_units, dig_tbm_specs,
                                                     dig_buf_params, cap_in_search_params,
                                                     dut))
        results = await helper.gather_err()
        cdict = {}
        for idx, pin in enumerate(pin_names):
            # extract data cap and max of all ctrl pin caps
            cin = results[idx]
            if pin == 'din':
                cdict['data'] = cin
            else:
                c_cur = cdict.get('ctrl', -float('inf'))
                if cin > c_cur:
                    cdict['ctrl'] = cin
        return cdict

    async def _get_inv_input_cap(self, pinfo: Any, vdd: float, seg: int,
                                 tbm_specs, buf_params, search_params) -> float:
        """
        Return the input cap of an inverter with `seg` segments using simulation
        """
        cload = seg*1e-15
        buf_params['inv_params'][0]['seg'] = int(np.round(seg/16))
        buf_params['inv_params'][1]['seg'] = int(np.round(seg/4))
        dut_params = dict(cls_name='bag3_digital.layout.stdcells.gates.InvCore', draw_taps=True,
                          params=dict(pinfo=pinfo, seg=seg))
        pwr_domain = {'in': ('VSS', 'VDD'), 'out': ('VSS', 'VDD')}
        sup_values = dict(VDD=vdd, VSS=0.0)
        pin_values = {}
        reset_list = []
        diff_list = []
        load_list = [dict(pin='out', type='cap', value=cload)]

        dut = await self.async_new_dut('cin_inv_dut', STDCellWrapper, dut_params)
        return await self._get_input_cap('in', dut, pwr_domain, sup_values, pin_values, reset_list,
                                         diff_list, tbm_specs, buf_params, search_params,
                                         load_list)

    async def _get_driver_input_cap(self, pin: str, dut_params: dict, vdd: float, cload: float,
                                    nsegs: int, tbm_specs, buf_params, search_params, dut) -> float:
        """
        Return the input cap of the output driver
        """

        driver_unit_info = dut_params['params']['unit_params']
        seg_nand = driver_unit_info['seg_nand']
        seg_nor = driver_unit_info['seg_nor']
        if pin == 'din':
            driver_segs_tot = nsegs * (seg_nand + seg_nor)
        elif pin == 'n_enb_drv<0>':
            driver_segs_tot = seg_nor
        elif pin == 'n_enb_drv<1>':
            driver_segs_tot = 2 * seg_nor
        elif pin == 'tristate':
            driver_segs_tot = 3 * seg_nor
        elif pin == 'p_en_drv<0>':
            driver_segs_tot = seg_nand
        elif pin == 'p_en_drv<1>':
            driver_segs_tot = 2 * seg_nand
        elif pin == 'tristateb':
            driver_segs_tot = 3 * seg_nand
        else:
            driver_segs_tot = dut_params['params']['pupd_params']['stack']

        inv1_segs = int(np.round(driver_segs_tot/16))
        inv1_segs = 1 if inv1_segs == 0 else inv1_segs
        inv2_segs = int(np.round(driver_segs_tot/4))
        inv2_segs = 1 if inv2_segs == 0 else inv2_segs
        buf_params['inv_params'][0]['seg'] = inv1_segs
        buf_params['inv_params'][1]['seg'] = inv2_segs

        sup_tuple = ('VSS', 'VDD')
        pwr_domain = {'din': sup_tuple, 'txpadout': sup_tuple, 'n_enb_drv': sup_tuple,
                      'p_en_drv': sup_tuple, 'tristate': sup_tuple, 'tristateb': sup_tuple,
                      'weak_puenb': sup_tuple, 'weak_pden': sup_tuple}
        sup_values = dict(VDD=vdd, VSS=0.0)
        if pin == 'din':
            pin_values = {'n_enb_drv<1:0>': 6-nsegs, 'p_en_drv<1:0>': nsegs-3,
                          'tristate': 0, 'trsistateb': 1, 'weak_puenb': 1, 'weak_pden': 0}
        elif pin.startswith('n_enb_drv') or pin == 'tristate':
            pin_values = {'din': 0}
        elif pin.startswith('p_en_drv') or pin == 'tristateb':
            pin_values = {'din': 1}
        elif pin == 'weak_puenb':
            pin_values = {'n_enb_drv<1:0>': 3, 'p_en_drv<1:0>': 0, 'tristate': 1, 'tristateb': 0,
                          'weak_pden': 0}
        elif pin == 'weak_pden':
            pin_values = {'n_enb_drv<1:0>': 3, 'p_en_drv<1:0>': 0, 'tristate': 1, 'tristateb': 0,
                          'weak_puenb': 1}
        else:
            raise self.error('pin name not valid')

        reset_list = []
        diff_list = []
        load_list = [dict(pin='txpadout', type='cap', value=cload)]

        return await self._get_input_cap(pin, dut, pwr_domain, sup_values, pin_values, reset_list,
                                         diff_list, tbm_specs, buf_params, search_params,
                                         load_list)

    async def _get_input_cap(self, in_pin, dut, pwr_domain, sup_values, pin_values, reset_list,
                             diff_list, tbm_specs, buf_params, search_params,
                             load_list) -> float:
        tbm_specs = dict(pwr_domain=pwr_domain, sup_values=sup_values, pin_values=pin_values,
                         reset_list=reset_list, diff_list=diff_list, **tbm_specs)
        mm_specs = dict(tbm_specs=tbm_specs, in_pin=in_pin, buf_params=buf_params,
                        search_params=search_params, load_list=load_list)
        mm = self.make_mm(CapDelayMatch, mm_specs)
        in_pin_dir = in_pin.replace('<', '_').replace('>', '_')
        mm_results = await self.async_simulate_mm_obj(f'{dut.cell_name}_{in_pin_dir}', dut, mm)
        data = mm_results.data
        cap_rise = data['cap_rise']
        cap_fall = data['cap_fall']
        return (cap_rise + cap_fall) / 2

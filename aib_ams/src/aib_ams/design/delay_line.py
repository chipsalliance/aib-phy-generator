# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
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

"""
Design Script for the NAND-based delay line. The design procedure was ported from
design/delay_line.py to support new testbench/measurement manager API and also DesignerBase.
The delay requirement should be relaxed on cds_ff_mpt to around 60ps so that the loops converge
quickly.
"""

from typing import Mapping, Dict, Any, Tuple, Optional, cast, Type

import numpy as np
from copy import deepcopy

from bag.layout.template import TemplateBase
from bag.simulation.design import DesignerBase
from bag.concurrent.util import GatherHelper

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag.layout.util import IPMarginTemplate
from xbase.layout.mos.top import GenericWrapper
from aib_ams.layout.delay_line import DelayLine

from bag.env import get_tech_global_info


class DelayLineDesigner(DesignerBase):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)
        self.dsn_tech_info = get_tech_global_info('aib_ams')

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return DelayLine

    async def async_design(self,
                           tmin: float,
                           tmax: float,
                           cload: float,
                           ncodes: int,
                           std_delay: float,
                           pinfo: Mapping[str, Any],
                           sim_params: Mapping[str, Any],
                           max_nand_seg: int,
                           max_stack: int,
                           nrows: int,
                           ncols: int,
                           mc_dsn_env: str,
                           mc_params: Mapping[str, Any],
                           **kwargs: Any) -> Mapping[str, Any]:
        """ Designs the delay cell.
        1) if td_step > tmax up-size outer nands.
        2) if td_step < tmin use higher vth or stack transistors to increase the resistance.
        3) if 2 did not work try up-sizing the inner nands.
        4) check condition 1, if it holds move to 5, otherwise repeat by re-sizing outer nands
        with the new inner nands
        5) if std(td_step) > std_targ, upsize all devices by (std(td_stp) / std_targ) ** 2.
        Parameters
        ----------
        tmin: float
            td_step should be greater than this spec.
        tmax: float
            td_step should be less than this spec.
        cload: float
            Loading capacitor.
        ncodes: int
            Total number of codes (the number of actual delay cells used).
        std_delay: float
            The standard deviation of td_step should be less than this spec across PVT.
        pinfo: Mapping[str, Any]
            Pinfo object.
        sim_params: Mapping[str, Any]
            Simulation parameter Dictionary. It should contain the following parameters:
            sim_envs: List[str]
                Corner/temp used for minimal simulations
            tper: float
                Time period of input signal
            ncycles: int
                The number cycles used for measurement of delay
            trf: float
                Rise/Fall time
            tclk: float
                The clk time period for latches to load to register the code
            clk_trf: float
                Clk signal rise/fall time
            thres_lo: float
                Low threshold used for delay measurements
            thres_hi: float
                High threshold used for delay measurements
        max_nand_seg: int
            Maximum number of segments to be used in nand gates.
        max_stack: int
            Maximum number of stacks to be used.
        nrows: int
            Number of rows for the matrix of delay cells in layout. The total number of cells
            computed by nrows x ncols should be more than the number of codes. The remaining
            un-used cells will be dummies.
        ncols: int
            Number of columns for the matrix of delay cells in layout.
        mc_dsn_env: str
            The monte-carlo environment name. Should be consistent with global tech vars.
        mc_params: Mapping[str, Any]
            Dictionary for monte-carlo parameters, it must include:
                numruns: int, seed: int, donominal: str, variations: str.
                numruns: 30, seed: 1, donominal: 'yes', variations: 'all' is recommended.
        kwargs: Any
            num_core: int = 1
                The number of cores in each delay cell. If one wants to use more stages per cell.
        Returns
        -------
        summary: Mapping[str, Any]
            Design summary.
        """
        tech_info = self.dsn_tech_info
        curr_nand_outer_seg = curr_nand_inner_seg = tech_info['seg_min']
        curr_stack = 1
        num_core = kwargs.get('num_core', 1)

        plot_result: bool = kwargs.get('plot_result', False)
        gen_specs: Optional[Mapping[str, Any]] = kwargs.get('gen_cell_specs', None)
        gen_cell_args: Optional[Mapping[str, Any]] = kwargs.get('gen_cell_args', None)

        cur_tmax, cur_tmin = float('inf'), -float('inf')
        while cur_tmax > tmax or cur_tmin < tmin:
            # Accomplish max delay by sweeping outer_nand_nseg from current to max possible
            self.log('First deal with max delay')
            if cur_tmax > tmax:
                for curr_nand_outer_seg in range(curr_nand_outer_seg, max_nand_seg + 1):
                    cur_tmax, cur_tmin = await self._design_chain_step(curr_nand_outer_seg,
                                                                       curr_nand_inner_seg,
                                                                       curr_stack,
                                                                       ncodes,
                                                                       cload,
                                                                       num_core,
                                                                       nrows,
                                                                       ncols,
                                                                       pinfo,
                                                                       sim_params)
                    self.log(f'cur_tmax = {cur_tmax}, wanted_tmax < {tmax}')
                    # await self.plot_results(curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload, num_core,
                    #     nrows, ncols, pinfo, sim_params)
                    if cur_tmax < tmax:
                        self.log(f'nand_out_seg = {curr_nand_outer_seg} worked')
                        break
                else:
                    raise ValueError("Unable to design cells to meet max delay spec")
            # Accomplish min delay by sweeping stack from current to max possible. In case of
            # failure to find a working size try up-sizing the inner_nand_nseg from current to max.
            # Since we are up-sizing the loading of outer nands, the tmax may get violated in which
            # case we have to upsize the outer nand and repeat the steps.
            self.log('Now deal with min delay')
            if cur_tmin < tmin:
                for curr_stack in range(curr_stack + 1, max_stack + 1):
                    cur_tmax, cur_tmin = await self._design_chain_step(curr_nand_outer_seg,
                                                                       curr_nand_inner_seg,
                                                                       curr_stack,
                                                                       ncodes,
                                                                       cload,
                                                                       num_core,
                                                                       nrows,
                                                                       ncols,
                                                                       pinfo,
                                                                       sim_params)
                    self.log(f'cur_tmin = {cur_tmin}, wanted_tmin > {tmin}')
                    # await self.plot_results(curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload,
                    #                         num_core,
                    #                         nrows, ncols, pinfo, sim_params)
                    if cur_tmin > tmin:
                        self.log('done with tmin using stack')
                        break
                else:
                    self.log('min delay not met with stacking; try increasing SR NAND number '
                             'of fingers')
                    # min delay not met with stacking; try increasing SR NAND number of fingers
                    for curr_nand_inner_seg in range(curr_nand_inner_seg + 1, max_nand_seg + 1):
                        cur_tmax, cur_tmin = await self._design_chain_step(curr_nand_outer_seg,
                                                                           curr_nand_inner_seg,
                                                                           curr_stack,
                                                                           ncodes,
                                                                           cload,
                                                                           num_core,
                                                                           nrows,
                                                                           ncols,
                                                                           pinfo,
                                                                           sim_params)
                        self.log(f'cur_tmin = {cur_tmin}, wanted_tmin > {tmin}')
                        # await self.plot_results(curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload,
                        #                         num_core,
                        #                         nrows, ncols, pinfo, sim_params)
                        if cur_tmin > tmin:
                            break
                    else:
                        raise ValueError("Unable to design cells to meet min delay spec")

        # for monte-carlo up-size everything until variation gets small.
        if mc_dsn_env:
            curr_nand_outer_seg, curr_nand_inner_seg = await self._design_for_mc(
                curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, cload, std_delay, num_core,
                pinfo, sim_params, mc_dsn_env, mc_params, ncodes=3)

        if plot_result:
            results = await self._measure_times(
                curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload, num_core,
                nrows, ncols, pinfo, sim_params
            )
            td_min = results['td_min']
            td_max = results['td_max']
            # results:
            # return tdr_arr, tdf_arr, tdr_per_code, tdf_per_code
            dsn_env_names = sim_params['sim_envs']
            from matplotlib import pyplot as plt
            plt.figure(1)
            ax: Any = plt.subplot(2, 1, 1)
            xvec = np.arange(0, ncodes / 2 - 1)
            for idx, sim_env in enumerate(dsn_env_names):
                tdr = results['tdr_arr'][idx, :-1]
                tdr = tdr[..., ::2].flatten()
                plt.step(xvec, tdr, where='mid', label=sim_env)
            ax.legend()
            ax.set_ylabel('Rise Delay (s)')
            ax = plt.subplot(2, 1, 2)
            for idx, sim_env in enumerate(dsn_env_names):
                tdr_step = results['tdr_per_code'][idx, :].flatten()
                ax.scatter(xvec, tdr_step, label=sim_env)
            ax.set_ylim(ymin=td_min, ymax=td_max)
            ax.legend()
            ax.set_ylabel('Rise Delay Step (s)')
            ax.set_xlabel('Code')
            plt.show()

        summary = dict(
            nand_outer_seg=curr_nand_outer_seg,
            nand_inner_seg=curr_nand_inner_seg,
            stack=curr_stack,
            num_insts=ncodes,
        )
        if gen_specs is not None and gen_cell_args is not None:
            dut_params = self._get_dut_params(curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, num_core,
                                              nrows, ncols, pinfo)
            # dut_params['params']['draw_taps'] = 'LEFT'
            gen_cell_specs = dict(
                lay_class=STDCellWrapper.get_qualified_name(),
                cls_name=DelayLine.get_qualified_name(),
                params=dut_params,
                **gen_specs,
            )
            return dict(gen_specs=gen_cell_specs, gen_args=gen_cell_args)

        return summary

    async def plot_results(self, curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload, num_core,
                           nrows, ncols, pinfo, sim_params):
        results = await self._measure_times(
            curr_nand_outer_seg, curr_nand_inner_seg, curr_stack, ncodes, cload, num_core,
            nrows, ncols, pinfo, sim_params
        )
        td_min = results['td_min']
        td_max = results['td_max']
        # results:
        # return tdr_arr, tdf_arr, tdr_per_code, tdf_per_code
        dsn_env_names = sim_params['sim_envs']
        from matplotlib import pyplot as plt
        plt.figure(1)
        ax: Any = plt.subplot(2, 1, 1)
        xvec = np.arange(0, ncodes / 2 - 1)
        for idx, sim_env in enumerate(dsn_env_names):
            tdr = results['tdr_arr'][idx, :-1]
            tdr = tdr[..., ::2].flatten()
            plt.step(xvec, tdr, where='mid', label=sim_env)
        ax.legend()
        ax.set_ylabel('Rise Delay (s)')
        ax = plt.subplot(2, 1, 2)
        for idx, sim_env in enumerate(dsn_env_names):
            tdr_step = results['tdr_per_code'][idx, :].flatten()
            ax.scatter(xvec, tdr_step, label=sim_env)
        ax.set_ylim(ymin=td_min, ymax=td_max)
        ax.legend()
        ax.set_ylabel('Rise Delay Step (s)')
        ax.set_xlabel('Code')
        plt.show()

    async def _measure_times(self,
                             nand_outer_seg: int,
                             nand_inner_seg: int,
                             stack: int,
                             ncodes: int,
                             cload: float,
                             num_core: int,
                             nrows: int,
                             ncols: int,
                             pinfo: Mapping[str, Any],
                             sim_params: Mapping[str, Any]):
        dut_params = self._get_dut_params(nand_outer_seg, nand_inner_seg, stack, ncodes, num_core,
                                          nrows, ncols, pinfo)
        dut = await self.async_new_dut('dly_line_chain', STDCellWrapper, dut_params)



        tper = sim_params['tper']
        ncycles = sim_params['ncycles']
        sim_id_pref = f'dly_no{nand_outer_seg}_ni{nand_inner_seg}_s{stack}'
        helper = GatherHelper()
        for code in range(0, ncodes - 1):
            tbm_params = self._get_tbm_params(code, ncodes, cload, sim_params)
            sim_id = f'{sim_id_pref}_c{code}'
            helper.append(self._get_delay(sim_id, dut, tbm_params, t_start=tper * (ncycles - 1),
                                          t_stop=tper * ncycles))
        ret_list = await helper.gather_err()
        tdr_list, tdf_list = [x[0] for x in ret_list], [x[1] for x in ret_list]

        res_arr = dict()

        tdr_arr = np.stack(tdr_list, axis=-1)
        res_arr['tdr_arr'] = tdr_arr
        tdf_arr = np.stack(tdf_list, axis=-1)
        res_arr['tdf_arr'] = tdf_arr
        tdr_per_code = np.diff(tdr_arr, axis=-1)[..., ::2]
        res_arr['tdr_per_code'] = tdr_per_code
        tdf_per_code = np.diff(tdf_arr, axis=-1)[..., ::2]
        res_arr['tdf_per_code'] = tdf_per_code

        tdr_max = np.max(tdr_per_code)
        res_arr['tdr_max'] = tdr_max
        tdf_max = np.max(tdf_per_code)
        res_arr['tdf_max'] = tdf_max
        tdr_min = np.min(tdr_per_code)
        res_arr['tdr_min'] = tdr_min
        tdf_min = np.min(tdf_per_code)
        res_arr['tdf_min'] = tdf_min

        res_arr['td_max'] = max(tdr_max, tdf_max)
        res_arr['td_min'] = min(tdr_min, tdf_min)

        return res_arr

    async def _design_chain_step(self,
                                 nand_outer_seg: int,
                                 nand_inner_seg: int,
                                 stack: int,
                                 ncodes: int,
                                 cload: float,
                                 num_core: int,
                                 nrows: int,
                                 ncols: int,
                                 pinfo: Mapping[str, Any],
                                 sim_params: Mapping[str, Any]) -> Tuple[float, float]:
        mid = ncodes // 2
        dut_params = self._get_dut_params(nand_outer_seg, nand_inner_seg, stack, ncodes, num_core,
                                          nrows, ncols, pinfo)
        # dut = await self.async_new_dut('dly_line_chain', STDCellWrapper, dut_params)
        dut = await self.async_new_dut('dly_line_chain', GenericWrapper, dut_params)

        tper = sim_params['tper']
        ncycles = sim_params['ncycles']
        sim_id_pref = f'dly_no{nand_outer_seg}_ni{nand_inner_seg}_s{stack}'
        helper = GatherHelper()
        for code in [0, 1, mid, mid + 1, ncodes - 2, ncodes - 1]:
            tbm_params = self._get_tbm_params(code, ncodes, cload, sim_params)
            sim_id = f'{sim_id_pref}_c{code}'
            helper.append(self._get_delay(sim_id, dut, tbm_params, t_start=tper * (ncycles - 1),
                                          t_stop=tper * ncycles))
        ret_list = await helper.gather_err()
        tdr_list, tdf_list = [x[0] for x in ret_list], [x[1] for x in ret_list]

        tdr_arr = np.stack(tdr_list, axis=-1)
        tdf_arr = np.stack(tdf_list, axis=-1)
        tdr_per_code = np.diff(tdr_arr, axis=-1)[..., ::2]
        tdf_per_code = np.diff(tdf_arr, axis=-1)[..., ::2]

        tdr_max = np.max(tdr_per_code)
        tdf_max = np.max(tdf_per_code)
        tdr_min = np.min(tdr_per_code)
        tdf_min = np.min(tdf_per_code)

        td_max = max(tdr_max, tdf_max)
        td_min = min(tdr_min, tdf_min)

        return td_max, td_min

    async def _design_for_mc(self,
                             outer_seg: int,
                             inner_seg: int,
                             stack: int,
                             cload: float,
                             std_dev_max: float,
                             num_core: int,
                             pinfo: Mapping[str, Any],
                             sim_params: Mapping[str, Any],
                             mc_dsn_env: str,
                             mc_params: Mapping[str, Any],
                             ncodes: int,
                             max_iter: int = 5
                             ) -> Tuple[int, int]:
        # ncodes in this part can be a small value (i.e. 3)
        global_info = self.dsn_tech_info
        indx = 0
        cur_std_max = float('inf')
        n_factor = 1
        tstart = sim_params['tper'] * (sim_params['ncycles'] - 1)
        tstop = sim_params['tper'] * sim_params['ncycles']
        # finish if maximum number of iteration reached
        while cur_std_max > std_dev_max:
            if indx > max_iter:
                raise ValueError(f'Reached the maximum niter {max_iter}, but have not reached '
                                 f'std spec')
            _factor = int(np.ceil(outer_seg * n_factor)) / outer_seg
            outer_seg = int(np.ceil(outer_seg * _factor))
            inner_seg = int(np.ceil(inner_seg * _factor))
            cload *= _factor
            self.log(f'Computed fudge factor: {n_factor}, actual factor: {_factor}')
            self.log(f'New outer NAND Seg: {outer_seg}')
            self.log(f'New inner NAND Seg: {inner_seg}')
            self.log(f'New cload: {cload}')

            # put all cells in a single row
            dut_params = self._get_dut_params(outer_seg, inner_seg, stack,
                                              ncodes=ncodes,
                                              num_core=num_core,
                                              nrows=1,
                                              ncols=ncodes + 2,
                                              pinfo=pinfo)
            dut = await self.async_new_dut('dly_line_chain', STDCellWrapper, dut_params)
            helper = GatherHelper()
            for code in range(ncodes):
                mc_sim_params = dict(deepcopy(sim_params))
                mc_sim_params['sim_envs'] = global_info['dsn_envs'][mc_dsn_env]['mc_env']
                mc_tb_params = self._get_tbm_params(code, ncodes, cload, mc_sim_params)
                mc_tb_params['monte_carlo_params'] = mc_params
                sim_id = f'dly_no{outer_seg}_ni{inner_seg}_s{stack}_mc{indx}_c{code}'
                self.log("Running Monte Carlo on design")
                helper.append(self._get_delay(sim_id, dut, mc_tb_params, t_start=tstart,
                                              t_stop=tstop))
            ret_list = await helper.gather_err()
            tdr_list, tdf_list = [x[0] for x in ret_list], [x[1] for x in ret_list]

            tdr = np.stack(tdr_list, axis=0)
            tdf = np.stack(tdf_list, axis=0)

            steps_r = np.diff(tdr, axis=0).squeeze()
            steps_f = np.diff(tdf, axis=0).squeeze()

            sdr = np.std(steps_r)
            sdf = np.std(steps_f)

            cur_std_max = max(sdr, sdf)
            print('=' * 80)
            self.log(f"Standard deviation of the rising delay: {sdr}, Spec = {std_dev_max}")
            self.log(f"Standard deviation of the falling delay: {sdf}, Spec = {std_dev_max}")
            print('=' * 80)

            diff_factor = cur_std_max / std_dev_max
            n_factor = diff_factor ** 2
            indx += 1

        return outer_seg, inner_seg

    def _get_dut_params(self,
                        nand_outer_seg: int,
                        nand_inner_seg: int,
                        stack: int,
                        ncodes: int,
                        num_core: int,
                        nrows: int,
                        ncols: int,
                        pinfo: Mapping[str, Any],
                        ) -> Dict[str, Any]:

        tech_info = self.dsn_tech_info
        dc_core = {
            'in': nand_outer_seg,
            'out': nand_outer_seg,
            'sr': nand_inner_seg,
        }

        scan_rst_flop = {
            'in': tech_info['seg_min'],
            'buf': tech_info['seg_min'],
            'keep': tech_info['seg_min'],
            'pass': tech_info['seg_min'],
            'mux': tech_info['seg_min'],
            'rst': tech_info['seg_min'],
            'out': tech_info['seg_min'],
        }

        dut_params = dict(
            cls_name=self.get_dut_lay_class().get_qualified_name(),
            params=dict(
                pinfo=pinfo,
                seg_dict=dict(
                    dc_core=dc_core,
                    scan_rst_flop=scan_rst_flop,
                    so_inv=tech_info['seg_min'],
                    bk_inv=tech_info['seg_min'],
                ),
                stack_nand=stack,
                num_rows=nrows,
                num_cols=ncols,
                num_insts=ncodes,
                num_core=num_core,
                flop=False,

                tile0=1,
                tile1=3,
                tile_vss=0,
                tile_vdd=2,
                substrate_row=True,
                draw_taps='NONE',
                show_pins=True,
            )
        )
        return dut_params

    def _get_tbm_params(self,
                        code: int,
                        num_inst: int,
                        cload: float,
                        sim_params: Mapping[str, Any],
                        ) -> Dict[str, Any]:
        tech_info = self.dsn_tech_info

        dut_pins = ['dlyin', 'CLKIN', 'VDD', 'VSS', 'iSE', 'RSTb',
                    'dlyout', 'iSI', 'SOOUT', f'a{num_inst - 1}',
                    f'b{num_inst - 1}', f'bk<{num_inst - 1}:0>']

        pwr_domain = {}
        for pin in dut_pins:
            if '<' in pin:
                base_name = pin.split('<')[0]
                pwr_domain[base_name] = ('VSS', 'VDD')
            else:
                pwr_domain[pin] = ('VSS', 'VDD')

        pulse_list = [
            dict(
                pin='dlyin',
                tper=sim_params['tper'],
                tpw=sim_params['tper'] / 2,
                trf=sim_params['trf'],
            ),
            dict(
                pin='CLKIN',
                tper=sim_params['tclk'],
                tpw=sim_params['tclk'] / 2,
                trf=sim_params['clk_trf'],
            )
        ]

        default_sim_env = tech_info['dsn_envs']['center']['env']
        sim_envs = sim_params.get('sim_envs', default_sim_env)
        tbm_specs = dict(
            sim_envs=sim_envs,
            save_outputs=['dlyin', 'dlyout'],
            sim_params=dict(
                t_sim=sim_params['ncycles'] * sim_params['tper'],
                t_rst=0,
                t_rst_rf=0,
            ),
            dut_pins=dut_pins,
            pulse_list=pulse_list,
            load_list=[dict(pin='dlyout', type='cap', value=cload)],
            pwr_domain=pwr_domain,
            sup_values=dict(VSS=0, VDD=tech_info['vdd']),
            pin_values={
                f'bk<{num_inst - 1}:0>': (1 << num_inst) - 1 - 2 ** code,
                'iSE': 0,
                'iSI': 0,
                'RSTb': 1,
                f'a{num_inst - 1}': 'mid',
                f'b{num_inst - 1}': 'mid',
            },
            thres_lo=sim_params['thres_lo'],
            thres_hi=sim_params['thres_hi']
        )

        return tbm_specs

    async def _get_delay(self,
                         sim_id: str,
                         dut,
                         tbm_specs: Mapping[str, Any],
                         **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

        tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
        sim_results = await self.async_simulate_tbm_obj(sim_id, dut, tbm, tbm_specs)
        tdr = tbm.calc_delay(sim_results.data, 'dlyin', 'dlyout',
                             in_edge=EdgeType.RISE, out_edge=EdgeType.RISE, **kwargs)
        tdf = tbm.calc_delay(sim_results.data, 'dlyin', 'dlyout',
                             in_edge=EdgeType.FALL, out_edge=EdgeType.FALL, **kwargs)
        return tdr, tdf

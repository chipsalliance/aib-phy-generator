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
from typing import Mapping, Dict, Any, Tuple, Sequence, Optional, cast

import copy
import pprint

import numpy as np

from aib_ams.layout.delay_line import DelayCellCore
from bag.concurrent.util import GatherHelper
from bag.env import get_tech_global_info
from bag.simulation.cache import DesignInstance
from bag.simulation.design import DesignerBase
from bag.util.immutable import ImmutableSortedDict, Param
from bag.util.search import BinaryIterator
from bag.layout.util import IPMarginTemplate

from aib_ams.layout.phase_interp import PhaseInterpolatorWithDelay
from bag3_analog.layout.phase.phase_interp import PhaseInterpolator
from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_liberty.data import parse_cdba_name
from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.digital import DigitalTranTB

from xbase.layout.mos.top import GenericWrapper


class PhaseInterpDesigner(DesignerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_info = get_tech_global_info('aib_ams')
        lch = self.global_info['lch_min']
        w_p = self.global_info['w_minp']
        w_n = self.global_info['w_minn']
        th = self.global_info['thresholds'][0]
        self._tech_dsn_base_params = dict(lch=lch, w_p=w_p, w_n=w_n, th_p=th, th_n=th)

    async def async_design(self, pinfo: Mapping[str, Any], nbits: int,
                           rtol: float, atol: float, tbit: float, trf: float, cload: float,
                           mc_params: Param, num_cores: int, target: Mapping[str, Any],
                           delay_cell_params: Mapping[str, Any], **kwargs: Mapping[str, Any]
                           ) -> Mapping[str, Any]:
        td_min = target['td_min']
        td_max = target['td_max']
        t_max = target['t_max']
        td_sigma = target['td_sigma']
        tristate_seg = kwargs.get('tristate_seg', self.global_info['seg_min'])
        tristate_stack = kwargs.get('tristate_stack', 1)
        seg_buf_abs_max = kwargs.get('seg_buf_abs_max', 50)
        seg_buf_max_override = kwargs.get('seg_buf_max_override', None)
        seg_buf_min_override = kwargs.get('seg_buf_min_override', None)
        design_using_signoff = kwargs.get('design_using_signoff', False)
        mc_corner = kwargs.get('mc_corner', 'tt_25')
        mc_env_override = kwargs.get("mc_env_override", None)
        mc_worst_corner = kwargs.get("mc_worst_corner", True)
        plot_result: bool = kwargs.get('plot_result', False)
        dsn_monte_carlo: bool = kwargs.get('dsn_monte_carlo', True)
        gen_specs: Optional[Mapping[str, Any]] = kwargs.get('gen_cell_specs', None)
        gen_cell_args: Optional[Mapping[str, Any]] = kwargs.get('gen_cell_args', None)

        # 0. Setup design environments and the testbench manager
        if design_using_signoff:
            dsn_envs = self.global_info['signoff_envs']
            dsn_env_names = dsn_envs['all_corners']['envs']
            dsn_env_vdds = dsn_envs['all_corners']['vdd']
        else:
            dsn_envs = self.global_info['dsn_envs']
            dsn_env_names = [env for dct in dsn_envs.values() for env in dct['env']]
            dsn_env_vdds = {e: dsn_envs[c]['vdd'] for c in dsn_envs.keys()
                            for e in dsn_envs[c]['env']}

        if not mc_worst_corner and not mc_env_override:
            raise ValueError("If not performing mc on the worst corner, specify mc_env_override!")

        dut_pins = ['a_in', 'b_in', 'intout', 'VDD', 'VSS', f'sp<{nbits - 1}:0>',
                    f'sn<{nbits - 1}:0>', 'a_in_buf']
        tbm_dict = {}
        for dsn_env in dsn_env_names:
            tbm_specs = self._get_tbm_specs([dsn_env], dict(vdd={dsn_env: dsn_env_vdds[dsn_env]}),
                                            dut_pins, tbit, trf, cload, nbits, rtol, atol)
            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
            tbm_dict[dsn_env] = tbm
        tbm_params = dict()

        # 1. Setup the base phase interpolator and extract the input cap
        pwr_domains = dict(b_in=('VSS', 'VDD'), a_in=('VSS', 'VDD'), out=('VSS', 'VDD'))
        cin_dut_conns = dict(a_in=1)
        for i in range(nbits):
            for name, value in zip(['a_en', 'b_en', 'a_enb', 'b_enb'], [1, 0, 0, 1]):
                cin_dut_conns[f'{name}<{i}>'] = value
                pwr_domains[f'{name}<{i}>'] = ('VSS', 'VDD')
        gen_params = dict(
            cls_name=PhaseInterpolator.get_qualified_name(),
            params=dict(
                pinfo=pinfo,
                unit_params={'seg': tristate_seg, 'stack_p': tristate_stack,
                             'stack_n': tristate_stack},
                inv_params={'seg': 2, 'stack_p': 1, 'stack_n': 1},
                nbits=nbits,
                draw_sub=True,
                export_outb=True,
                abut_tristates=True,
            )
        )
        pi_in_cap = await self._get_input_cap(gen_params, 'b_in', cload, cin_dut_conns, pwr_domains,
                                              [dict(pin='out', type='cap', value='c_out')],
                                              vdd=dsn_env_vdds['tt_25'], sim_envs=['tt_25'])
        # 2. Setup the base delay cell and extract it's input cap
        pin_names = ['bk1', 'ci_p', 'co_p', 'in_p', 'out_p']
        pwr_domains = {pin_name: ('VSS', 'VDD') for pin_name in pin_names}
        cin_dut_conns = dict(bk1=1, ci_p=0)
        load_list = [dict(pin='out_p', type='cap', value='c_out'),
                     dict(pin='co_p', type='cap', value='c_out')]
        gen_params = dict(
            cls_name=DelayCellCore.get_qualified_name(),
            params=dict(pinfo=pinfo, **delay_cell_params)
        )
        dc_in_cap = await self._get_input_cap(gen_params, 'ci_p', cload, cin_dut_conns, pwr_domains,
                                              load_list, vdd=dsn_env_vdds['tt_25'],
                                              sim_envs=['tt_25'])

        # 3. Size the delay cell to be able to drive the phase interpolator
        dc_up_scale_factor = int(round(pi_in_cap / dc_in_cap))
        delay_cell_params['seg_dict']['in'] *= dc_up_scale_factor
        delay_cell_params['seg_dict']['out'] *= dc_up_scale_factor
        delay_cell_params['seg_dict']['sr'] *= dc_up_scale_factor
        nand_seg = delay_cell_params['seg_dict']['out'] * 2

        inv_in_cap = self.global_info['cin_inv']['cin_per_seg']
        inv_seg = int(round(np.sqrt(dc_in_cap * pi_in_cap) / inv_in_cap))

        # 4. Upsize the buffer inverter on the output
        self.log('-' * 80)
        if seg_buf_min_override:
            self.log(f"Minimum Segments Overridden to {seg_buf_min_override}")
            min_seg_buf = seg_buf_min_override
        else:
            self.log('Find the min size for all the codes to be positive')
            seg_buf_min_iter = BinaryIterator(2, None, 2)
            while seg_buf_min_iter.has_next():
                _seg_buf = seg_buf_min_iter.get_next()
                dut_params = self._update_dut_params(pinfo, nbits, tristate_seg, seg_buf=_seg_buf,
                                                     seg_inv=inv_seg, seg_nand=nand_seg,
                                                     num_cores=num_cores,
                                                     dc_params=delay_cell_params)
                results = await self._measure_times(dsn_env_names, tbm_dict, dut_params, tbm_params,
                                                    tbit, nbits, name=f'sim_min_{_seg_buf}')
                # find min and max delay step
                tstep_min = results['min_step']
                tstep_max = results['max_step']
                self.log(f"Got min delay {tstep_min}, max delay {tstep_max}, with "
                         f"{_seg_buf} segments")
                if tstep_min < 0:
                    seg_buf_min_iter.up()
                else:
                    seg_buf_min_iter.save()
                    seg_buf_min_iter.down()
            min_seg_buf = seg_buf_min_iter.get_last_save()

        self.log('-' * 80)
        if seg_buf_max_override:
            self.log(f'Maximum Segments Overridden to {seg_buf_max_override}')
            max_seg_buf = seg_buf_max_override
        else:
            self.log('Now find the maximum size for all the codes to be positive')
            seg_buf_max_iter = BinaryIterator(10, None, 2)
            max_reached = False
            while seg_buf_max_iter.has_next():
                _seg_buf = seg_buf_max_iter.get_next()
                dut_params = self._update_dut_params(pinfo, nbits, tristate_seg, seg_buf=_seg_buf,
                                                     seg_inv=inv_seg, seg_nand=nand_seg,
                                                     num_cores=num_cores,
                                                     dc_params=delay_cell_params)
                results = await self._measure_times(dsn_env_names, tbm_dict, dut_params, tbm_params,
                                                    tbit, nbits, name=f'sim_max_{_seg_buf}')
                # find min and max delay step
                tstep_min = results['min_step']
                tstep_max = results['max_step']
                self.log(f"Got min delay {tstep_min}, max delay {tstep_max}, with "
                         f"{_seg_buf} segments")
                if tstep_min < 0:
                    seg_buf_max_iter.down()
                elif _seg_buf > seg_buf_abs_max:
                    max_reached = True
                    break
                else:
                    seg_buf_max_iter.save()
                    seg_buf_max_iter.up()
            max_seg_buf = seg_buf_max_iter.get_last_save() if not max_reached else seg_buf_abs_max
        self.log('-' * 80)
        self.log(f'Minimum Buffer segments to keep positive delays: {min_seg_buf}')
        self.log(f'Maximum Buffer segments to keep positive delays: {max_seg_buf}')

        seg_buf_bin_iter = BinaryIterator(min_seg_buf, max_seg_buf, 2)
        while seg_buf_bin_iter.has_next():
            _seg_buf = seg_buf_bin_iter.get_next()
            dut_params = self._update_dut_params(pinfo, nbits, tristate_seg, seg_buf=_seg_buf,
                                                 seg_inv=inv_seg, seg_nand=nand_seg,
                                                 num_cores=num_cores, dc_params=delay_cell_params)
            results = await self._measure_times(dsn_env_names, tbm_dict, dut_params, tbm_params,
                                                tbit, nbits, 'sim_size')
            tdelay_max = results['max_dly']
            tstep_min = results['min_step']
            tstep_max = results['max_step']
            if tdelay_max > t_max and tstep_min > td_min and tstep_max < td_max:
                # delay constraint violated, linearity constraint met
                seg_buf_bin_iter.down()
            elif tdelay_max < t_max and (tstep_min < td_min or tstep_max > td_max):
                # delay constraint met, linearity constraint violated
                seg_buf_bin_iter.up()
            elif tdelay_max < t_max and tstep_min > td_min and tstep_max < td_max:
                # both constraints met
                seg_buf_bin_iter.save_info((dut_params, results))
                seg_buf_bin_iter.down()
            else:
                self.error('Both delay and linearity constraints violated, please relax specs.')

        seg_buf_final = seg_buf_bin_iter.get_last_save()
        if not seg_buf_final:
            self.error("Design failed!, unable to meet linearity specs within range of inv sizes")
        self.log(f'Final output buffer size is {seg_buf_final}, before Monte Carlo sim.')
        dut_params, results = seg_buf_bin_iter.get_last_save_info()

        if dsn_monte_carlo:
            # 5. Monte Carlo simulations
            mc_tbm_dict = {}
            if mc_worst_corner:
                mc_envs = [mc_corner]
                mc_vdd = dict(vdd={mc_corner: dsn_env_vdds[mc_corner]})
                mc_tbm_specs = self._get_tbm_specs([mc_corner], mc_vdd, dut_pins, tbit, trf, cload,
                                                   nbits, rtol, atol)
                mc_tbm_specs['monte_carlo_params'] = mc_params
                mc_tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, mc_tbm_specs))
                mc_tbm_dict[mc_corner] = mc_tbm
            else:
                # TODO
                mc_envs = ...
                ...

            dut_params = self._update_dut_params(pinfo, nbits, tristate_seg, seg_buf=seg_buf_final,
                                                 seg_inv=inv_seg, seg_nand=nand_seg,
                                                 num_cores=num_cores, dc_params=delay_cell_params)

            mc_results = await self._measure_times(mc_envs, mc_tbm_dict, dut_params, tbm_params,
                                                   tbit, nbits, name='sim_mc_pre')
            mc_factor, sigma_max = self._get_mc_factor(mc_results, td_sigma)
            self.log(f'Max std. dev. is {sigma_max}')
            self.log(f'Upscale everything by {mc_factor}')
            self.log('-' * 80)

            # 6. Final verification
            seg_unit_final = int(np.ceil(tristate_seg * mc_factor))
            seg_unit_final += seg_unit_final & 1  # layout constraint
            seg_buf_final = int(np.ceil(seg_buf_final * mc_factor))
            seg_buf_final += seg_buf_final & 1  # layout constraint
            delay_cell_params_scale = copy.deepcopy(delay_cell_params)
            for key in delay_cell_params['seg_dict']:
                delay_cell_params_scale['seg_dict'][key] = int(
                    np.ceil(delay_cell_params['seg_dict'][key] * mc_factor))
            nand_seg = int(np.ceil(nand_seg * mc_factor))
            inv_seg = int(np.ceil(inv_seg * mc_factor))
            dut_params = self._update_dut_params(pinfo, nbits, seg_unit_final,
                                                 seg_buf=seg_buf_final, seg_inv=inv_seg,
                                                 seg_nand=nand_seg, num_cores=num_cores,
                                                 dc_params=delay_cell_params_scale)

            results = await self._measure_times(dsn_env_names, tbm_dict, dut_params, tbm_params,
                                                tbit, nbits, name='sim_sized')
            mc_results = await self._measure_times(mc_envs, mc_tbm_dict, dut_params, tbm_params,
                                                   tbit, nbits, name='sim_mc_post')
            _, sigma_max = self._get_mc_factor(mc_results, td_sigma)
            self.log(f'Final Sigma: {sigma_max}')
            self.log('-' * 80)
        else:
            seg_unit_final = tristate_seg
            delay_cell_params_scale = delay_cell_params

        self.log('-' * 80)
        self.log(f'dsn_envs: {dsn_env_names}')
        self.log(f'final results:\n{pprint.pformat(results, width=100)}')

        if plot_result:
            from matplotlib import pyplot as plt
            plt.figure(1)
            ax: Any = plt.subplot(2, 1, 1)
            xvec = np.arange(0, results['tdr_step'].shape[1])
            for idx, sim_env in enumerate(dsn_env_names):
                tdr = results['tdrs'][idx, :-1].flatten()
                plt.step(xvec, tdr, where='mid', label=sim_env)
            ax.legend()
            ax.set_ylabel('Rise Delay (s)')
            ax = plt.subplot(2, 1, 2)
            for idx, sim_env in enumerate(dsn_env_names):
                tdr_step = results['tdr_step'][idx, :].flatten()
                ax.scatter(xvec, tdr_step, label=sim_env)
            ax.set_ylim(ymin=td_min, ymax=td_max)
            ax.legend()
            ax.set_ylabel('Rise Delay Step (s)')
            ax.set_xlabel('Code')
            plt.show()

        if gen_specs is not None and gen_cell_args is not None:
            gen_cell_specs = dict(
                lay_class=IPMarginTemplate.get_qualified_name(),
                params=dict(
                    cls_name=GenericWrapper.get_qualified_name(),
                    params=dict(
                        cls_name=PhaseInterpolatorWithDelay.get_qualified_name(),
                        params=dut_params,
                    ),
                ),
                **gen_specs,
            )
            return dict(gen_specs=gen_cell_specs, gen_args=gen_cell_args)

        return dict(
            seg_unit=seg_unit_final,
            seg_buf=seg_buf_final,
            seg_dc=delay_cell_params_scale['seg_dict'],
            nand_seg=nand_seg,
            inv_seg=inv_seg,
        )

    @staticmethod
    def _get_mc_factor(results: Dict[str, Any], td_sigma: float) -> Tuple[float, float]:
        tdr_step = results['tdr_step']
        tdf_step = results['tdf_step']
        tdr_dc = results['tdrs_dc']
        tdf_dc = results['tdfs_dc']
        var_r = np.var(tdr_step, axis=-1)
        var_f = np.var(tdf_step, axis=-1)
        # Correct the variance of the last code
        var_dcr = np.var(tdr_dc, axis=-1)
        var_dcf = np.var(tdf_dc, axis=-1)
        # TODO: figure out a better implementation for the following nested loop
        cov_r = np.zeros(var_dcr.shape)
        cov_f = np.zeros(var_dcf.shape)
        for cor_idx in range(tdr_dc.shape[0]):
            for trf_idx in range(tdr_dc.shape[1]):
                cov_r = np.cov(tdr_step[cor_idx, -1, trf_idx], tdr_dc[cor_idx, trf_idx])[0, 1]
                cov_f = np.cov(tdf_step[cor_idx, -1, trf_idx], tdf_dc[cor_idx, trf_idx])[0, 1]
        var_r[:, -1] += var_dcr - 2 * cov_r
        var_f[:, -1] += var_dcf - 2 * cov_f
        sigma_r = np.sqrt(var_r)
        sigma_f = np.sqrt(var_f)
        sigma_max = max(np.max(sigma_r), np.max(sigma_f))
        mc_upscale_factor = (sigma_max / td_sigma) ** 2
        return max(1, mc_upscale_factor), sigma_max

    async def _measure_times(self, sim_envs: Sequence[str], tbm_dict: Dict[str, DigitalTranTB],
                             dut_params: Param, tbm_params: Mapping[str, Any], tbit: float,
                             nbits: int, name: str) -> Dict[str, Any]:
        gen_params = dict(
            cls_name=PhaseInterpolatorWithDelay.get_qualified_name(),
            params=dut_params,
        )
        dut = await self.async_new_dut('phase_interp', GenericWrapper, gen_params)

        helper = GatherHelper()
        for corner in sim_envs:
            helper.append(self._measure_times_at_corner(tbm_dict[corner], dut, tbm_params, tbit,
                                                        nbits, f'{name}_{corner}'))
        results = await helper.gather_err()

        dct = {k: [] for k in ['tdrs', 'tdfs', 'tdrs_dc', 'tdfs_dc', 'tdr_step', 'tdf_step']}
        max_steps, min_steps, max_dly = [], [], []
        for idx, corner in enumerate(sim_envs):
            tdr, tdf, tdr_dc, tdf_dc = results[idx]
            max_dly.append(max(np.min(tdr), np.min(tdf)))
            tdr = np.vstack((tdr, (tdr[0] + tdr_dc)[None, ...]))
            tdf = np.vstack((tdf, (tdf[0] + tdf_dc)[None, ...]))
            tdr_step = np.diff(tdr, axis=0)
            tdf_step = np.diff(tdf, axis=0)
            dct['tdr_step'].append(tdr_step)
            dct['tdf_step'].append(tdf_step)
            dct['tdrs'].append(tdr)
            dct['tdfs'].append(tdf)
            dct['tdrs_dc'].append(tdr_dc)
            dct['tdfs_dc'].append(tdf_dc)
            max_steps.append(max(np.max(tdr_step), np.max(tdf_step)))
            min_steps.append(min(np.min(tdr_step), np.min(tdf_step)))
        for k in ['tdrs', 'tdfs', 'tdrs_dc', 'tdfs_dc', 'tdr_step', 'tdf_step']:
            dct[k] = np.array(dct[k])
        max_idx = cast(int, np.argmax(np.array(max_steps)))
        min_idx = cast(int, np.argmin(np.array(min_steps)))
        dct['max_corner'] = sim_envs[max_idx]
        dct['min_corner'] = sim_envs[min_idx]
        dct['max_step'] = max_steps[max_idx]
        dct['min_step'] = min_steps[min_idx]
        dct['max_dly'] = max(max_dly)
        return dct

    async def _measure_times_at_corner(self, tbm: DigitalTranTB, dut: DesignInstance,
                                       tbm_params: Mapping[str, Any], tbit: float, nbits: int,
                                       name: str
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sim_results = await self.async_simulate_tbm_obj(name, dut, tbm, tbm_params)
        tdrs, tdfs, tdrs_dc, tdfs_dc = [], [], [], []
        for i in range(0, nbits + 1):
            tdrs.append(tbm.calc_delay(sim_results.data, 'a_in', 'intout', in_edge=EdgeType.RISE,
                                       out_edge=EdgeType.RISE, t_start=(2 * i + 1 + 0.25) * tbit,
                                       t_stop=(2 * i + 1 + 1.25) * tbit))
            tdfs.append(tbm.calc_delay(sim_results.data, 'a_in', 'intout', in_edge=EdgeType.FALL,
                                       out_edge=EdgeType.FALL, t_start=(2 * i + 1 + 0.5) * tbit,
                                       t_stop=(2 * i + 1 + 2) * tbit))
            tdrs_dc.append(tbm.calc_delay(sim_results.data, 'a_in_buf', 'b_in',
                                          in_edge=EdgeType.RISE, out_edge=EdgeType.RISE,
                                          t_start=(2 * i + 1 + 0.25) * tbit,
                                          t_stop=(2 * i + 1 + 1.5) * tbit))
            tdfs_dc.append(tbm.calc_delay(sim_results.data, 'a_in_buf', 'b_in',
                                          in_edge=EdgeType.FALL, out_edge=EdgeType.FALL,
                                          t_start=(2 * i + 1 + 0.25) * tbit,
                                          t_stop=(2 * i + 1 + 1.5) * tbit))
        return np.array(tdrs), np.array(tdfs), np.mean(tdrs_dc, axis=0), np.mean(tdfs_dc, axis=0)

    async def _get_input_cap(self, gen_params: Mapping[str, Any], in_pin: str, cload: float,
                             pin_values: Mapping[str, int],
                             pwr_domain: Mapping[str, Tuple[str, str]],
                             load_list: Sequence[Mapping[str, str]],
                             sim_envs: Sequence[str], vdd: float,
                             seg_inv_1: int = 2, seg_inv_2: int = 8) -> float:
        # First setup base inverter params from technology specs
        inv_params_base = self._tech_dsn_base_params
        bparams = dict(inv_params=[dict(seg=seg_inv_1, **inv_params_base),
                                   dict(seg=seg_inv_2, **inv_params_base)],
                       export_pins=True)

        # Set the tbm_specs
        sim_params = dict(t_bit=1.0e-9, t_rst=0, t_rst_rf=0, t_rf=self.global_info['trf_nom'],
                          c_out=cload)
        sup_values = dict(VDD=vdd, VSS=0.0)
        tbm_specs = dict(sim_envs=sim_envs, sim_params=sim_params, pwr_domain=pwr_domain,
                         sup_values=sup_values, pin_values=pin_values, reset_list=[], diff_list=[],
                         thres_lo=0.1, thres_hi=0.9, rtol=1.0e-9, atol=1.0e-22,
                         tran_options=dict(maxstep=1.0e-12, errpreset='conservative'))

        # Set the meas params
        search_params = dict(low=1.0e-18, high=None, step=1.0e-16, tol=1.0e-17, max_err=1.0e-9,
                             overhead_factor=5)
        meas_params = dict(tbm_specs=tbm_specs,
                           in_pin=in_pin,
                           buf_params=bparams,
                           search_params=search_params,
                           load_list=load_list)

        dut = await self.async_new_dut('cin_inv_dut', GenericWrapper, gen_params)
        mm = self.make_mm(CapDelayMatch, meas_params)
        mm_results = await self.async_simulate_mm_obj(f'sim_obj', dut, mm)
        data = mm_results.data
        return (data['cap_rise'] + data['cap_fall']) / 2

    @staticmethod
    def _get_tbm_specs(sim_envs: Sequence[str], env_params: Mapping[str, Any],
                       dut_pins: Sequence[str], tbit: float, trf: float, cload: float, nbits: int,
                       rtol: float, atol: float
                       ) -> Dict[str, Any]:
        tsim = tbit * (2 * nbits + 2) + tbit / 2
        pulse_list = [dict(pin='a_in', tper=tbit, tpw=tbit / 2, trf=trf, td=tbit / 2)]
        for i in range(nbits):
            pulse_list.append(dict(pin=f'sn<{i}>', tper=2 * tsim, tpw=tsim, trf=trf,
                                   td=(2 * i + 2) * tbit + tbit / 4, pos=False))
            pulse_list.append(dict(pin=f'sp<{i}>', tper=2 * tsim, tpw=tsim, trf=trf,
                                   td=(2 * i + 2) * tbit + tbit / 4, pos=True))

        pin_values = {}
        load_list = [dict(pin='intout', type='cap', value=cload)]
        pwr_domains = {parse_cdba_name(pin)[0]: ('VSS', 'VDD') for pin in dut_pins}
        sim_params = dict(
            t_sim=tsim,
            t_rst=0,
            t_rst_rf=trf,
        )
        sup_values = dict(VSS=0, VDD=env_params['vdd'])
        return dict(
            sim_params=sim_params,
            dut_pins=dut_pins,
            pulse_list=pulse_list,
            load_list=load_list,
            pwr_domain=pwr_domains,
            sup_values=sup_values,
            pin_values=pin_values,
            reset_list=[],
            diff_list=[],
            rtol=rtol,
            atol=atol,
            sim_envs=sim_envs,
            env_params=env_params,
            save_outputs=['a_in', 'a_in_buf', 'intout', 'b_in']
        )

    @staticmethod
    def _update_dut_params(pinfo: Mapping[str, Any], nbits: int, seg_unit: int, seg_buf: int,
                           seg_nand: int, seg_inv: int, num_cores: int,
                           dc_params: Mapping[str, Any]) -> Param:
        return ImmutableSortedDict(dict(
            pinfo=pinfo,
            pi_params=dict(
                unit_params={'seg': seg_unit, 'stack_p': 1, 'stack_n': 1},
                inv_params={'seg': seg_buf, 'stack_p': 1, 'stack_n': 1},
                abut_tristates=True,
            ),
            dc_params=dc_params,
            inv_params=dict(seg=seg_inv),
            nand_params=dict(seg=seg_nand),
            num_core=num_cores,
            nbits=nbits,
            export_dc_out=True,
            export_dc_in=True,
            draw_sub=True,
            # export_outb=True,
        ))

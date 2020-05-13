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

from typing import Dict, Any, Tuple, Optional, Union, Mapping

import pprint
from pathlib import Path

import numpy as np
from scipy.stats import linregress

from bag.io.file import write_yaml
from bag.concurrent.util import GatherHelper
from bag.simulation.core import TestbenchManager
from bag.simulation.cache import DesignInstance, SimulationDB, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from ..tran.digital import DigitalTranTB
from .comb import CombLogicTimingMM


class RCDelayCharMM(MeasurementManager):
    """Characterize delay of a digital gate using pulse sources in series with a resistor.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : Union[str, Sequence[str]]
        input pin(s)
    out_pin : Union[str, Sequence[str]]
        output pin(s)
    out_invert : Union[bool, Sequence[bool]]
        True if output is inverted from input.  Corresponds to each input/output pair.
    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            t_rf :
                input rise/fall time
    r_src : float
        nominal source resistance.
    c_load : float
        nominal load capacitance.
    scale_min : float
        lower bound scale factor for c_load/r_src.
    scale_max : float
        upper bound scale factor for c_load/r_src.
    num_samples : int
        number of data points to measure.
    c_in : float
        Defaults to 0.  If nonzero, add this input capacitance.
    wait_cycles : int
        Defaults to 0.  Number of cycles to wait toggle before finally measuring delay.
    t_step_min : float
        Defaults to 0.1e-12 (0.1 ps).  rise/fall time of the step function approximation.
    plot : bool
        Defaults to False.  True to plot fitted lines.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cin_specs: Mapping[str, Any] = {}
        self._td_specs: Mapping[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        out_invert: bool = specs['out_invert']
        tbm_specs: Mapping[str, Any] = specs['tbm_specs']
        r_src: float = specs['r_src']
        c_load: float = specs['c_load']
        scale_min: float = specs['scale_min']
        scale_max: float = specs['scale_max']
        num_samples: int = specs['num_samples']
        c_in: float = specs.get('c_in', 0)
        wait_cycles: int = specs.get('wait_cycles', 0)
        t_step_min: float = specs.get('t_step_min', 0.1e-12)

        cin_tbm_specs = dict(**tbm_specs)
        cin_tbm_specs['sim_params'] = sim_params = dict(**cin_tbm_specs['sim_params'])
        sim_params['r_src'] = r_src
        sim_params['c_load'] = c_load
        sim_params['t_rf'] = t_step_min
        sim_params['t_bit'] = f'10*(r_src+{r_src:.4g})*(c_load+{c_load:.4g})'
        cin_tbm_specs['swp_info'] = [('r_src', dict(type='LOG', start=r_src * scale_min,
                                                    stop=r_src * scale_max, num=num_samples))]

        if c_in:
            sim_params['c_in'] = c_in
            load_list = [dict(pin=in_pin, type='cap', value='c_in')]
        else:
            load_list = None

        self._cin_specs = dict(
            in_pin=in_pin,
            out_pin=out_pin,
            out_invert=False,
            tbm_specs=cin_tbm_specs,
            start_pin=DigitalTranTB.get_r_src_pin(in_pin),
            stop_pin=in_pin,
            out_rise=True,
            out_fall=True,
            wait_cycles=wait_cycles,
            add_src_res=True,
            load_list=load_list,
        )

        td_tbm_specs = cin_tbm_specs.copy()
        td_tbm_specs['swp_info'] = [('c_load', dict(type='LOG', start=c_load * scale_min,
                                                    stop=c_load * scale_max, num=num_samples))]

        self._td_specs = dict(
            in_pin=in_pin,
            out_pin=out_pin,
            out_invert=out_invert,
            tbm_specs=td_tbm_specs,
            out_rise=True,
            out_fall=True,
            wait_cycles=wait_cycles,
            add_src_res=True,
            load_list=load_list,
        )

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        c_in: float = specs.get('c_in', 0)
        t_step_min: float = specs.get('t_step_min', 0.1e-12)
        plot: bool = specs.get('plot', False)

        cin_mm = sim_db.make_mm(CombLogicTimingMM, self._cin_specs)
        td_mm = sim_db.make_mm(CombLogicTimingMM, self._td_specs)

        gatherer = GatherHelper()
        gatherer.append(sim_db.async_simulate_mm_obj(f'{name}_cin', sim_dir / 'cin', dut, cin_mm))
        gatherer.append(sim_db.async_simulate_mm_obj(f'{name}_td', sim_dir / 'td', dut, td_mm))
        cin_output, td_output = await gatherer.gather_err()

        t_unit = 10 * t_step_min
        sim_envs = cin_output.data['sim_envs']
        r_src = cin_output.data['sim_params']['r_src']
        cin_timing = cin_output.data['timing_data'][in_pin]
        td_cin_rise = cin_timing['cell_rise']
        td_cin_fall = cin_timing['cell_fall']
        rin_rise, cin_rise = self.fit_rc_in(td_cin_rise, r_src, c_in, t_unit)
        rin_fall, cin_fall = self.fit_rc_in(td_cin_fall, r_src, c_in, t_unit)

        td_timing = td_output.data['timing_data'][out_pin]
        c_load = td_output.data['sim_params']['c_load']
        td_out_rise = td_timing['cell_rise']
        td_out_fall = td_timing['cell_fall']
        rout_rise, cout_rise = self.fit_rc_out(td_out_rise, c_load, t_unit)
        rout_fall, cout_fall = self.fit_rc_out(td_out_fall, c_load, t_unit)

        if plot:
            from matplotlib import pyplot as plt
            if len(sim_envs) > 100:
                raise ValueError('Can only plot with num. sim_envs < 100')
            for idx in range(len(sim_envs)):
                ri_r = rin_rise[idx]
                ri_f = rin_fall[idx]
                ci_r = cin_rise[idx]
                ci_f = cin_fall[idx]
                ro_r = rout_rise[idx]
                ro_f = rout_fall[idx]
                co_r = cout_rise[idx]
                co_f = cout_fall[idx]

                rs = r_src[idx, ...]
                cl = c_load[idx, ...]
                td_cin_r = td_cin_rise[idx, ...]
                td_cin_f = td_cin_fall[idx, ...]
                td_out_r = td_out_rise[idx, ...]
                td_out_f = td_out_fall[idx, ...]

                plt.figure(idx * 100 + 1)
                plt.title(f'{sim_envs[idx]} c_in')
                plt.plot(rs, td_cin_r, 'bo', label='td_rise')
                plt.plot(rs, td_cin_f, 'ko', label='td_fall')
                plt.plot(rs, np.log(2) * rs * (ci_r + c_in) + ri_r * ci_r, '-r',
                         label='td_rise_fit')
                plt.plot(rs, np.log(2) * rs * (ci_f + c_in) + ri_f * ci_f, '-g',
                         label='td_fall_fit')
                plt.legend()

                plt.figure(idx * 100 + 2)
                plt.title(f'{sim_envs[idx]} rc_out')
                plt.plot(cl, td_out_r, 'bo', label='td_rise')
                plt.plot(cl, td_out_f, 'ko', label='td_fall')
                plt.plot(cl, ro_r * (co_r + cl), '-r', label='td_rise_fit')
                plt.plot(cl, ro_f * (co_f + cl), '-g', label='td_fall_fit')
                plt.legend()

            plt.show()

        ans = dict(
            sim_envs=sim_envs,
            r_in=(rin_fall, rin_rise),
            c_in=(cin_fall, cin_rise),
            c_out=(cout_fall, cout_rise),
            r_out=(rout_fall, rout_rise),
        )

        self.log(f'Measurement {name} done, result:\n{pprint.pformat(ans)}')
        write_yaml(sim_dir / f'{name}.yaml', ans)
        return ans

    @classmethod
    def fit_rc_in(cls, td: np.ndarray, rin_linear: np.ndarray, c_in_const: float, t_unit: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
        # Note: delay = Rs * (Ci + Cp) + Ri*Ci
        num_corners = td.shape[0]
        rvec = np.empty(num_corners)
        cvec = np.empty(num_corners)
        for idx in range(num_corners):
            c0, t0, _, _, _ = linregress(np.log(2) * rin_linear[idx, :],
                                         td[idx, :] / t_unit)
            ci = c0 - (c_in_const / t_unit)
            ri = t0 / ci
            ci *= t_unit
            rvec[idx] = ri
            cvec[idx] = ci
        return rvec, cvec

    @classmethod
    def fit_rc_out(cls, td: np.ndarray, cl: np.ndarray, t_unit: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
        num_corners = td.shape[0]
        rvec = np.empty(num_corners)
        cvec = np.empty(num_corners)
        for idx in range(num_corners):
            r0, t0, _, _, _ = linregress(cl[idx, :] / t_unit, td[idx, :] / t_unit)
            c0 = t0 * t_unit / r0
            rvec[idx] = r0
            cvec[idx] = c0
        return rvec, cvec

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

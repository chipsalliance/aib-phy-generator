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

from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import DesignInstance, SimulationDB, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM


class PassGateRCDelayCharMM(MeasurementManager):
    """Characterize RC of a passgate.

    Notes
    -----
    specification dictionary has the following entries:

    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            t_rf :
                input rise/fall time.
    r_src : float
        nominal source resistance.
    c_load : float
        nominal load capacitance.
    scale_min : float
        lower bound scale factor for c_load/r_src.
    scale_max : float
        upper bound scale factor for c_load/r_src.
    num_samples : int
        number of data points to measure for r_src/c_load.
    c_in : float
        Defaults to 0.  If nonzero, add this input capacitance.
    wait_cycles : int
        Defaults to 0.  Number of cycles to wait toggle before finally measuring delay.
    t_step_min: float:
        Defaults to 0.1ps.  small step size used to approxmiate step function, also used to
        estimate time unit (time unit = 10 * t_step_min).
    plot : bool
        Defaults to False.  True to plot fitted lines.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._td_specs: Mapping[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        specs = self.specs
        tbm_specs_orig: Mapping[str, Any] = specs['tbm_specs']
        r_src: float = specs['r_src']
        c_load: float = specs['c_load']
        scale_min: float = specs['scale_min']
        scale_max: float = specs['scale_max']
        num_samples: int = specs['num_samples']
        c_in: float = specs.get('c_in', 0)
        wait_cycles: int = specs.get('wait_cycles', 0)
        t_step_min: float = specs.get('t_step_min', 0.1e-12)

        r_swp = dict(type='LOG', start=r_src * scale_min, stop=r_src * scale_max, num=num_samples)
        c_swp = dict(type='LOG', start=c_load * scale_min, stop=c_load * scale_max, num=num_samples)
        tbm_specs = dict(**tbm_specs_orig)
        tbm_specs['sim_params'] = sim_params = dict(**tbm_specs['sim_params'])
        sim_params['t_bit'] = f'10*(r_src+{r_src:.4g})*(c_load+{c_load:.4g})'
        sim_params['t_rf'] = t_step_min
        if c_in:
            sim_params['c_in'] = c_in
            load_list = [dict(pin='s', type='cap', value='c_in')]
        else:
            load_list = None

        tbm_specs['swp_info'] = [('r_src', r_swp), ('c_load', c_swp)]
        pwr_tup = ('VSS', 'VDD')
        tbm_specs['pwr_domain'] = {pin: pwr_tup for pin in ['en', 'enb', 's', 'd']}
        tbm_specs['pin_values'] = dict(en=1, enb=0)
        tbm_specs['reset_list'] = tbm_specs['diff_list'] = []

        self._td_specs = dict(
            in_pin='s',
            out_pin='d',
            out_invert=False,
            tbm_specs=tbm_specs,
            start_pin=DigitalTranTB.get_r_src_pin('s'),
            out_rise=True,
            out_fall=True,
            wait_cycles=wait_cycles,
            add_src_res=True,
            load_list=load_list,
        )

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        r_unit: float = specs['r_src']
        c_unit: float = specs['c_load']
        c_in: float = specs.get('c_in', 0)
        plot: bool = specs.get('plot', False)
        t_step_min: float = specs.get('t_step_min', 0.1e-12)

        td_mm = sim_db.make_mm(CombLogicTimingMM, self._td_specs)
        mm_output = await sim_db.async_simulate_mm_obj(f'{name}_td', sim_dir / 'td', dut, td_mm)
        mm_result = mm_output.data

        sim_envs = mm_result['sim_envs']
        sim_params = mm_result['sim_params']
        # NOTE: with ideal step input, the "delay resistance" is ln(2) times physical resistance.
        r_td = sim_params['r_src'] * np.log(2)
        c_load = sim_params['c_load']

        delay_data = mm_result['timing_data']['d']
        td_rise: np.ndarray = delay_data['cell_rise']
        td_fall: np.ndarray = delay_data['cell_fall']

        num = len(sim_envs)
        res_rise = np.empty(num)
        res_fall = np.empty(num)
        cs_rise = np.empty(num)
        cs_fall = np.empty(num)
        cd_rise = np.empty(num)
        cd_fall = np.empty(num)
        t_unit = 10 * t_step_min
        for idx in range(len(sim_envs)):
            rs = r_td[idx, ...]
            cl = c_load[idx, ...]
            tdr = td_rise[idx, ...]
            tdf = td_fall[idx, ...]
            self._fit_rc(idx, tdr, rs, c_in, cl, res_rise, cs_rise, cd_rise, r_unit, c_unit, t_unit)
            self._fit_rc(idx, tdf, rs, c_in, cl, res_fall, cs_fall, cd_fall, r_unit, c_unit, t_unit)

        if plot:
            # noinspection PyUnresolvedReferences
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import pyplot as plt
            from matplotlib import cm
            if len(sim_envs) > 100:
                raise ValueError('Can only plot with num. sim_envs < 100')
            for idx in range(len(sim_envs)):
                rs = r_td[idx, ...]
                cl = c_load[idx, ...]
                tdr = td_rise[idx, ...]
                tdf = td_fall[idx, ...]

                rs_fine = np.logspace(np.log10(rs[0]), np.log10(rs[-1]), num=51)
                cl_fine = np.logspace(np.log10(cl[0]), np.log10(cl[-1]), num=51)
                tdr_calc = (rs_fine * (cl_fine + cs_rise[idx] + cd_rise[idx]) +
                            res_rise[idx] * (cd_rise[idx] + cl_fine))
                tdf_calc = (rs_fine * (cl_fine + cs_fall[idx] + cd_fall[idx]) +
                            res_fall[idx] * (cd_fall[idx] + cl_fine))

                for fig_idx, rf_str, td, td_calc in [(idx * 100, 'rise', tdr, tdr_calc),
                                                     (idx * 100 + 1, 'fall', tdf, tdf_calc)]:
                    fig = plt.figure(fig_idx)
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_title(f'{sim_envs[idx]}_{rf_str}')
                    ax.plot_surface(rs_fine, cl_fine, td_calc, rstride=1, cstride=1,
                                    cmap=cm.get_cmap('cubehelix'))
                    ax.scatter(rs.flatten(), cl.flatten(), td.flatten(), c='k')
            plt.show()

        return dict(
            sim_envs=sim_envs,
            r_p=(res_fall, res_rise),
            c_s=(cs_fall, cs_rise),
            c_d=(cd_fall, cd_rise),
        )

    @staticmethod
    def _fit_rc(idx: int, td: np.ndarray, rs: np.ndarray, c_in: float, cl: np.ndarray,
                res: np.ndarray, cs: np.ndarray, cd: np.ndarray, r_unit: float, c_unit: float,
                t_unit: float) -> None:
        c_min = 1.0e-18

        rs_flat = rs.flatten()
        cl_flat = cl.flatten()
        a = np.empty((rs_flat.size, 3))
        a[:, 0] = rs_flat * c_unit / t_unit
        a[:, 1] = cl_flat * r_unit / t_unit
        a[:, 2] = 1

        b = (td.flatten() - rs_flat * (cl_flat + c_in)) / t_unit
        x = lstsq(a, b)[0]
        rp = x[1] * r_unit
        cd0 = x[2] * t_unit / rp
        cs0 = x[0] * c_unit - cd0
        if cs0 < 0 or cd0 < 0:
            # we got negative capacitance, which causes problems.
            # Now, assume the rp we got is correct, do another least square fit to enforce
            # cs and cd are positive
            a2 = np.empty((rs_flat.size, 2))
            a2[:, 0] = a[:, 0]
            a2[:, 1] = a[:, 0] + (rp * c_unit / t_unit)
            b -= rp * cl_flat / t_unit
            # noinspection PyUnresolvedReferences
            opt_res = lsq_linear(a2, b, bounds=(c_min, float('inf'))).x
            cs0 = opt_res[0] * c_unit
            cd0 = opt_res[1] * c_unit

        res[idx] = rp
        cs[idx] = cs0
        cd[idx] = cd0

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

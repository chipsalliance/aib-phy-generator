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

from typing import Any, Union, Tuple, Mapping, Type, Optional, Dict, Sequence, cast

from pathlib import Path

import numpy as np

from bag.math import float_to_si_string
from bag.util.importlib import import_class
from bag.concurrent.util import GatherHelper
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimData
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasInfo, MeasurementManager

from bag3_liberty.enum import TimingType

from ...data.tran import EdgeType
from ..util import setup_digital_tran
from .base import FlopMeasMode, FlopTimingBase
from .timing import FlopConstraintTimingMM


class FlopTimingCharMM(MeasurementManager):
    """Characterize all timing constraints of a flop.

    Assumes that timing margin is between [-t_clk_per/4, t_clk_per/4]

    Notes
    -----
    specification dictionary has the following entries:

    flop_params : Mapping[str, Any]
        flop parameters.
    delay_thres : float
        Defaults to 0.05.  Percent increase in delay for setup/hold constraints.  Use infinity
        to disable.  At least one of delay_thres or delay_inc must be specified.  If both are
        given, both constraints will be satisfied.
    delay_inc : float
        Defaults to infinity.  Increase in delay in seconds for setup/hold constraints.  At least
        one  delay_thres or delay_inc must be specified.  If both are given, both constraints
        will be satisfied.
    constraint_min_map : Mapping[Tuple[str, bool], float]
        mapping from measurement mode to min constraint map.
    sim_env_name : str
        Use to query for sim_env dependent timing offset.
    tbm_cls : Union[str, Type[FlopTimingBase]]
        The testbench class.
    tbm_specs : Mapping[str, Any]
        TestbenchManager specifications.
    c_load : float
        load capacitance for input constraint characterizations.
    t_rf_list : Sequence[float]
        list of input rise/fall time values for characterization.
    t_clk_rf_list : Sequence[float]
        list of clock rise/fall time values for input characterization.
    t_clk_rf_first : bool
        True if clock rise/fall time is the first axis of input characterization.
    out_swp_info : Sequence[Any]
        the swp_info object for output delay characterization.
    search_params : Mapping[str, Any]
        interval search parameters, with the following entries:

        max_margin : float
            Optional.  maximum timing margin in seconds.  Defaults to t_clk_per/4.
        tol : float
            tolerance of the binary search.  Terminate the search when it is below this value.
        overhead_factor : float
            ratio of simulation startup time to time it takes to simulate one sweep point.
    fake : bool
        Defaults to False.  True to output fake data for debugging.
    use_dut : bool
        Defaults to True.  True to instantiate DUT.
    wrapper_params : Mapping[str, Any]
        Used only if simulated with a DUT wrapper.  Contains the following entries:

        lib : str
            wrapper library name.
        cell : str
            wrapper cell name.
        params : Mapping[str, Any]
            DUT wrapper schematic parameters.
        pins : Sequence[str]
            wrapper pin list.
        power_domain : Mapping[str, Tuple[str, str]
            power domain of wrapper.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        flop_params: Mapping[str, Any] = specs['flop_params']
        tbm_cls_val: Union[str, Type[FlopTimingBase]] = specs['tbm_cls']
        t_rf_list: Sequence[float] = specs['t_rf_list']
        t_clk_rf_list: Sequence[float] = specs['t_clk_rf_list']
        t_clk_rf_first: bool = specs['t_clk_rf_first']

        fake: bool = specs.get('fake', False)
        use_dut: bool = specs.get('use_dut', True)

        meas_dut = dut if use_dut else None

        tbm_cls = cast(Type[FlopTimingBase], import_class(tbm_cls_val))

        mode_list = tbm_cls.get_meas_modes(flop_params)
        out_mode_list = tbm_cls.get_output_meas_modes(flop_params)

        # output timing measurement
        self.log('Measuring output delays')
        timing_table = {}
        gatherer = GatherHelper()
        for out_mode in out_mode_list:
            gatherer.append(self.get_out_timing(name, sim_db, meas_dut, sim_dir, tbm_cls,
                                                out_mode, fake, timing_table))

        self.log('Measuring input constraints')
        clk_len = len(t_clk_rf_list)
        in_len = len(t_rf_list)
        arr_shape = (clk_len, in_len) if t_clk_rf_first else (in_len, clk_len)
        for ck_idx, t_clk_rf in enumerate(t_clk_rf_list):
            for rf_idx, t_rf in enumerate(t_rf_list):
                for meas_mode in mode_list:
                    coro = self.get_in_timing(name, sim_db, meas_dut, sim_dir, meas_mode,
                                              fake, t_clk_rf, t_rf, ck_idx, rf_idx,
                                              arr_shape, t_clk_rf_first, timing_table)
                    gatherer.append(coro)

        await gatherer.gather_err()

        ans = {key: list(val.values()) for key, val in timing_table.items()}
        return ans

    async def get_in_timing(self, name: str, sim_db: SimulationDB, dut: Optional[DesignInstance],
                            sim_dir: Path, meas_mode: FlopMeasMode, fake: bool,
                            t_clk_rf: float, t_rf: float, ck_idx: int, rf_idx: int,
                            arr_shape: Tuple[int, ...], t_clk_rf_first: bool,
                            timing_table: Dict[str, Any]) -> None:
        constraint_min_map: Mapping[Tuple[str, bool], float] = self.specs.get('constraint_min_map',
                                                                              {})

        mm_specs = self.specs.copy()
        mm_specs.pop('constraint_min_map', None)
        mm_specs['meas_mode'] = meas_mode
        mm_specs['fake'] = fake
        cons_key = (meas_mode.input_mode_name, meas_mode.meas_setup)
        mm_specs['constraint_min'] = constraint_min_map.get(cons_key, float('-inf'))
        mm = sim_db.make_mm(FlopConstraintTimingMM, mm_specs)
        sim_params = mm.specs['tbm_specs']['sim_params']
        sim_params['t_clk_rf'] = t_clk_rf
        sim_params['t_rf'] = t_rf
        sim_params['c_load'] = mm_specs['c_load']
        mm.commit()

        state = self._get_state(meas_mode, t_clk_rf, t_rf, mm_specs['tbm_specs']['sim_envs'])

        meas_result = await sim_db.async_simulate_mm_obj(f'{name}_{state}', sim_dir / state,
                                                         dut, mm)

        meas_data = meas_result.data
        arr_sel = (ck_idx, rf_idx) if t_clk_rf_first else (rf_idx, ck_idx)
        for val_dict in meas_data.values():
            timing_info: Mapping[str, Any] = val_dict['timing_info']
            timing_val: float = val_dict['value']
            pin_data_list: Sequence[Tuple[str, str]] = timing_info['pin_data_list']
            ttype_str: str = timing_info['timing_type']

            for pin_name, data_name in pin_data_list:
                arr_table = _get_arr_table(timing_table, pin_name, ttype_str, timing_info)
                arr = arr_table.get(data_name, None)
                if arr is None:
                    arr_table[data_name] = arr = np.full(arr_shape, np.nan)
                arr[arr_sel] = timing_val

    async def get_out_timing(self, name: str, sim_db: SimulationDB, dut: Optional[DesignInstance],
                             sim_dir: Path, tbm_cls: Type[FlopTimingBase], meas_mode: FlopMeasMode,
                             fake: bool, timing_table: Dict[str, Any]) -> None:
        specs = self.specs
        search_params: Mapping[str, Any] = specs['search_params']
        flop_params: Mapping[str, Any] = specs['flop_params']
        out_swp_info: Sequence[Any] = specs['out_swp_info']
        sim_env_name: str = specs.get('sim_env_name', '')

        tbm_specs, tb_params = setup_digital_tran(specs, dut, meas_mode=meas_mode,
                                                  flop_params=flop_params,
                                                  sim_env_name=sim_env_name)
        tbm = cast(FlopTimingBase, sim_db.make_tbm(tbm_cls, tbm_specs))
        c_load = specs.get('c_load', None)
        if c_load is not None:
            tbm.sim_params['c_load'] = c_load
        if tbm.num_sim_envs != 1:
            raise ValueError('Cannot have corner sweep in flop characterization.')
        sim_id = f'out_delay_{meas_mode.name.lower()}_{tbm.sim_envs[0]}'

        tbm.set_swp_info(out_swp_info)
        output_map = tbm.get_output_map(True)
        ttype = TimingType.rising_edge if meas_mode.is_pos_edge_clk else TimingType.falling_edge
        ttype_str = ttype.name
        if fake:
            sim_data = None
        else:
            sim_params = tbm.sim_params
            t_clk_per = sim_params['t_clk_per']
            t_rf = sim_params['t_rf'] / (tbm.thres_hi - tbm.thres_lo)
            max_margin: float = search_params.get('max_margin', t_clk_per / 4)
            # NOTE: max_timing_value makes sure PWL width is always non-negative
            max_timing_value = max_margin + t_rf
            for var in tbm.timing_variables:
                sim_params[var] = max_timing_value

            sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id, dut, tbm,
                                                              tb_params, tb_name=f'{name}_{sim_id}')

            sim_data = sim_results.data

        # fill in results
        data_shape = tbm.sweep_shape[1:]
        for timing_info, edge_out_list in output_map.values():
            offset = timing_info.get('offset', 0)
            for edge, out_list in edge_out_list:
                for out_pin in out_list:
                    arr_table = _get_arr_table(timing_table, out_pin, ttype_str, timing_info)
                    td, trf = _get_out_data(tbm, sim_data, out_pin, edge, data_shape)
                    if edge is EdgeType.RISE:
                        arr_table['cell_rise'] = td + offset
                        arr_table['rise_transition'] = trf
                    else:
                        arr_table['cell_fall'] = td + offset
                        arr_table['fall_transition'] = trf

    def _get_state(self, mode: FlopMeasMode, t_ck_rf: float, t_rf: float, sim_envs: Sequence[str]
                   ) -> str:
        if len(sim_envs) != 1:
            raise ValueError('Only support single corner.')
        precision = self.precision
        ck_str = float_to_si_string(t_ck_rf, precision=precision).replace('.', 'd')
        rf_str = float_to_si_string(t_rf, precision=precision).replace('.', 'd')
        if mode.is_reset:
            state = f'{sim_envs[0]}_{mode.name}_ck_{ck_str}_rf_{rf_str}'
        else:
            state = (f'{sim_envs[0]}_{mode.name}_{"setup" if mode.meas_setup else "hold"}_'
                     f'ck_{ck_str}_rf_{rf_str}')
        return state

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')


class FlopTimingFakeMM(MeasurementManager):
    """Generate fake timing constraints of a flop.

    Notes
    -----
    specification dictionary has the following entries in addition to IntervalSearchMM:

    flop_params : Mapping[str, Any]
        map from pin name to timing constraint values.
    t_rf_list : Sequence[float]
        list of input rise/fall time values for characterization.
    t_clk_rf_list : Sequence[float]
        list of clock rise/fall time values for input characterization.
    t_clk_rf_first : bool
        True if clock rise/fall time is the first axis of input characterization.
    out_swp_info : Sequence[Any]
        the swp_info object for output delay characterization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        flop_params: Mapping[str, Any] = specs['flop_params']
        t_rf_list: Sequence[float] = specs['t_rf_list']
        t_clk_rf_list: Sequence[float] = specs['t_clk_rf_list']
        t_clk_rf_first: bool = specs['t_clk_rf_first']
        out_swp_info: Sequence[Any] = specs['out_swp_info']

        ans = {}
        clk_len = len(t_clk_rf_list)
        in_len = len(t_rf_list)
        in_shape = (clk_len, in_len) if t_clk_rf_first else (in_len, clk_len)
        out_shape = TestbenchManager.get_sweep_shape(1, out_swp_info)
        for pin_name, timing_list in flop_params.items():
            ans[pin_name] = new_timing = []
            for timing_data in timing_list:
                new_table = {k: timing_data[k] for k in ['cond', 'related', 'timing_type']}
                timing_type: str = new_table['timing_type']

                ttype: TimingType = TimingType[timing_type]
                data_shape = out_shape if ttype.is_output else in_shape
                new_table['data'] = {key: np.full(data_shape, val)
                                     for key, val in timing_data['data'].items()}

                new_timing.append(new_table)
        return ans

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')


def _get_arr_table(timing_table: Dict[str, Any], pin_name: str, ttype_str: str,
                   timing_info: Mapping[str, Any]) -> Dict[str, Any]:
    pin_timing_table = timing_table.get(pin_name, None)
    if pin_timing_table is None:
        timing_table[pin_name] = pin_timing_table = {}
    result_table = pin_timing_table.get(ttype_str, None)
    if result_table is None:
        arr_table = {}
        pin_timing_table[ttype_str] = dict(
            related=timing_info['related'],
            cond=timing_info['cond'],
            data=arr_table,
            timing_type=ttype_str,
        )
    else:
        arr_table = result_table['data']

    return arr_table


def _get_out_data(tbm: FlopTimingBase, data: Optional[SimData], out_pin: str, edge: EdgeType,
                  data_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    if data is None:
        td = np.full(data_shape, 50.0e-12)
        trf = np.full(data_shape, 20.0e-12)
    else:
        td = tbm.calc_clk_to_q(data, out_pin, edge)
        trf = tbm.calc_out_trf(data, out_pin, edge)
        if td.shape[0] == 1:
            td = td[0, ...]
            trf = trf[0, ...]

    return td, trf

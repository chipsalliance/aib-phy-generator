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

from typing import Any, Union, Tuple, Mapping, List, Optional, Dict, Sequence, Type, cast

from pathlib import Path

import numpy as np

from pybag.enum import TermType

from bag.util.immutable import update_recursive
from bag.concurrent.util import GatherHelper
from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasInfo, MeasurementManager

from bag3_liberty.enum import TimingSenseType, TimingType
from bag3_liberty.util import get_bus_bit_name, parse_cdba_name, cdba_to_unusal
from bag3_liberty.boolean import build_timing_cond_expr

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM
from bag3_testbenches.measurement.digital.flop.char import FlopTimingCharMM

from ..cap.delay_match import CapDelayMatch
from ..cap.max_trf import CapMaxRiseFallTime


class LibertyCharMM(MeasurementManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tran_specs: Mapping[str, Any] = {}
        self._cin_specs: Dict[str, Any] = {}
        self._cout_specs: Dict[str, Any] = {}
        self._delay_specs: Dict[str, Any] = {}
        self._seq_mm_table: Dict[str, MeasurementManager] = {}

        super().__init__(*args, **kwargs)

    @property
    def fake(self) -> bool:
        return self.specs.get('fake', False)

    def commit(self) -> None:
        specs = self.specs
        fake = self.fake

        sim_env_name: str = specs['sim_env_name']
        sim_envs: Sequence[str] = specs['sim_envs']
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        dut_info: Mapping[str, Any] = specs['dut_info']
        tran_tbm_specs: Mapping[str, Any] = specs['tran_tbm_specs']
        buf_params: Mapping[str, Any] = specs['buf_params']
        in_cap_search_params: Mapping[str, Any] = specs['in_cap_search_params']
        out_cap_search_params: Mapping[str, Any] = specs['out_cap_search_params']
        seq_search_params: Mapping[str, Any] = specs['seq_search_params']
        seq_delay_thres: float = specs['seq_delay_thres']
        seq_timing: Mapping[str, Mapping[str, Any]] = specs['seq_timing']
        t_rf_list: Sequence[float] = specs['t_rf_list']
        t_clk_rf_list: Sequence[float] = specs['t_clk_rf_list']
        t_clk_rf_first: bool = specs['t_clk_rf_first']

        delay_swp_info: Sequence[Any] = specs['delay_swp_info']
        seq_swp_info: Sequence[Any] = specs['seq_swp_info']

        cap_tbm_specs = dict(**tran_tbm_specs)
        cap_tbm_specs['sim_envs'] = sim_envs
        cap_tbm_specs['thres_lo'] = thres_lo
        cap_tbm_specs['thres_hi'] = thres_hi
        cap_tbm_specs.update(dut_info)
        self._cin_specs = dict(
            tbm_specs=cap_tbm_specs,
            in_pin='',
            buf_params=buf_params,
            search_params=in_cap_search_params,
        )

        self._cout_specs = dict(
            tbm_specs=cap_tbm_specs,
            in_pin='',
            out_pin='',
            max_trf=0,
            buf_params=buf_params,
            search_params=out_cap_search_params,
        )

        delay_tbm_specs = cap_tbm_specs.copy()
        delay_tbm_specs['swp_info'] = delay_swp_info
        self._tran_specs = delay_tbm_specs
        self._delay_specs = dict(
            tbm_specs=delay_tbm_specs,
            in_pin='',
            out_pin='',
            out_invert=False,
            fake=fake,
        )

        self._seq_mm_table.clear()
        for name, seq_timing_specs in seq_timing.items():
            mm_cls: Union[Type[MeasurementManager], str] = seq_timing_specs.get('mm_cls',
                                                                                FlopTimingCharMM)
            seq_specs = dict(
                delay_thres=seq_delay_thres,
                sim_env_name=sim_env_name,
                tbm_specs=cap_tbm_specs,
                t_rf_list=t_rf_list,
                t_clk_rf_list=t_clk_rf_list,
                t_clk_rf_first=t_clk_rf_first,
                out_swp_info=seq_swp_info,
                search_params=seq_search_params,
                fake=fake,
            )
            seq_specs.update(seq_timing_specs)
            self._seq_mm_table[name] = cast(MeasurementManager, self.make_mm(mm_cls, seq_specs))

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        in_cap_min: float = specs['in_cap_min_default']
        out_max_trf: float = specs['out_max_trf']
        out_min_fanout: float = specs['out_min_fanout']
        in_cap_table: Mapping[str, float] = specs['in_cap_table']
        out_io_info_table: Mapping[str, Mapping[str, Any]] = specs['out_io_info_table']
        custom_meas: Mapping[str, Mapping[str, Any]] = specs['custom_meas']

        # setup input capacitance measurements
        ans = {}
        out_io_pins = []
        gatherer = GatherHelper()
        for pin_name, term_type in dut.sch_master.pins.items():
            basename, bus_range = parse_cdba_name(pin_name)
            if bus_range is None:
                ans[pin_name] = pin_info = {}
                if term_type is TermType.input:
                    gatherer.append(self._measure_in_cap(name, sim_dir, sim_db, dut, pin_name,
                                                         in_cap_table, pin_info))
                else:
                    out_io_pins.append(pin_name)
            else:
                for bus_idx in bus_range:
                    bit_name = get_bus_bit_name(basename, bus_idx, cdba=True)
                    ans[bit_name] = pin_info = {}
                    if term_type is TermType.input:
                        gatherer.append(self._measure_in_cap(name, sim_dir, sim_db, dut,
                                                             bit_name, in_cap_table, pin_info))
                    else:
                        out_io_pins.append(bit_name)

        # record input capacitances
        if gatherer:
            in_cap_min = min((val for val in await gatherer.gather_err() if val is not None))

        # get parameters needed for output pin measurement
        out_cap_min = in_cap_min * out_min_fanout

        # compute inout and output pin cap/timing information
        gatherer.clear()
        for bit_name in out_io_pins:
            pin_info = out_io_info_table.get(bit_name, None)
            if pin_info is None:
                continue

            cap_info: Mapping[str, Any] = pin_info.get('cap_info', None)
            tinfo_list: Optional[Sequence[Mapping[str, Any]]] = pin_info.get('timing_info', None)

            output_table = ans[bit_name]
            if cap_info is not None:
                related: str = cap_info.get('related', '')
                max_cap: Optional[float] = cap_info.get('max_cap', None)
                max_trf: float = cap_info.get('max_trf', out_max_trf)
                cond: Mapping[str, int] = cap_info.get('cond', {})

                gatherer.append(self._measure_out_cap(name, sim_dir, sim_db, dut, bit_name, related,
                                                      max_cap, max_trf, cond, out_cap_min,
                                                      output_table))

            if tinfo_list is not None:
                output_table['timing'] = timing_output = []
                for idx, tinfo in enumerate(tinfo_list):
                    related: str = tinfo['related']
                    sense_str: str = tinfo['sense']
                    cond: Mapping[str, int] = tinfo.get('cond', {})
                    timing_type: str = tinfo.get('timing_type', 'combinational')
                    zero_delay: bool = tinfo.get('zero_delay', False)
                    data: Optional[Mapping[str, Any]] = tinfo.get('data', None)

                    related_str = cdba_to_unusal(related)
                    out_str = cdba_to_unusal(bit_name)
                    sim_id = f'comb_delay_{related_str}_{out_str}_{idx}'
                    gatherer.append(self._measure_delay(name, sim_id, sim_dir, sim_db, dut,
                                                        bit_name, related, sense_str, cond,
                                                        timing_type, zero_delay, data,
                                                        timing_output))
        # add custom and flop measurements
        for meas_name, meas_params in custom_meas.items():
            meas_cls: str = meas_params['meas_class']
            meas_specs: Mapping[str, Any] = meas_params['meas_specs']
            gatherer.append(self._measure_custom(name, sim_dir, sim_db, dut, meas_name,
                                                 meas_cls, meas_specs, ans))

        for seq_name, seq_mm in self._seq_mm_table.items():
            gatherer.append(self._measure_flop(name, sim_dir, sim_db, dut, seq_name, seq_mm, ans))

        # run all simulation in parallel
        await gatherer.run()
        return ans

    async def _measure_in_cap(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                              dut: Optional[DesignInstance], pin_name: str,
                              in_cap_table: Mapping[str, float], output_table: Dict[str, Any]
                              ) -> float:
        cap_range: float = self.specs['in_cap_range_scale']
        if self.fake:
            cap_rise = cap_fall = in_cap_table[pin_name]
        else:
            sim_id = f'cap_in_{cdba_to_unusal(pin_name)}'

            cur_specs = self._cin_specs.copy()
            cur_specs['in_pin'] = pin_name

            mm = sim_db.make_mm(CapDelayMatch, cur_specs)
            mm_result = await sim_db.async_simulate_mm_obj(f'{name}_{sim_id}', sim_dir / sim_id,
                                                           dut, mm)
            mm_data = mm_result.data
            cap_rise = mm_data['cap_rise']
            cap_fall = mm_data['cap_fall']

        cap = (cap_rise + cap_fall) / 2
        cap_rise_range = [cap_rise * (1 - cap_range), cap_rise * (1 + cap_range)]
        cap_fall_range = [cap_fall * (1 - cap_range), cap_fall * (1 + cap_range)]
        output_table['cap_dict'] = dict(cap=cap, cap_rise=cap_rise, cap_fall=cap_fall,
                                        cap_rise_range=cap_rise_range,
                                        cap_fall_range=cap_fall_range)
        return cap

    async def _measure_out_cap(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                               dut: Optional[DesignInstance], pin_name: str, related: str,
                               max_cap: Optional[float], max_trf: float, cond: Mapping[str, int],
                               out_cap_min: float, output_table: Dict[str, Any]) -> None:
        out_cap_num_freq: int = self.specs['out_cap_num_freq']

        if max_cap is None:
            if not related:
                raise ValueError('No related pin specified for max output cap measurement.')

            if self.fake:
                max_cap = 200.0e-15
            else:
                sim_id = f'cap_out_{cdba_to_unusal(pin_name)}'

                cur_specs = self._cout_specs.copy()
                cur_specs['in_pin'] = related
                cur_specs['out_pin'] = pin_name
                cur_specs['max_trf'] = max_trf
                if cond:
                    pin_values = cur_specs['tbm_specs']['pin_values'].copy()
                    pin_values.update(cond)
                    update_recursive(cur_specs, pin_values, 'tbm_specs', 'pin_values')

                mm = sim_db.make_mm(CapMaxRiseFallTime, cur_specs)
                mm_result = await sim_db.async_simulate_mm_obj(f'{name}_{sim_id}', sim_dir / sim_id,
                                                               dut, mm)
                mm_data = mm_result.data
                max_cap = mm_data['cap']

        output_table['cap_dict'] = dict(
            cap_min=min(max_cap, out_cap_min),
            cap_max=max_cap,
            cap_max_table=[max_cap] * out_cap_num_freq,
        )

    async def _measure_delay(self, name: str, sim_id: str, sim_dir: Path, sim_db: SimulationDB,
                             dut: Optional[DesignInstance], pin_name: str, related: str,
                             sense_str: str, cond: Mapping[str, int], timing_type_str: str,
                             zero_delay: bool, user_data: Optional[Mapping[str, Any]],
                             output_list: List[Dict[str, Any]]) -> None:
        specs = self.specs
        sim_env_name: str = specs['sim_env_name']
        delay_shape: Tuple[int, ...] = specs['delay_shape']

        sense = TimingSenseType[sense_str]
        if sense is TimingSenseType.non_unate:
            raise ValueError('Must specify timing sense for output measurement')
        out_invert = (sense is TimingSenseType.negative_unate)

        ttype: TimingType = TimingType[timing_type_str]
        keys = []
        if ttype.is_rising:
            keys.append('cell_rise')
            keys.append('rise_transition')
        if ttype.is_falling:
            keys.append('cell_fall')
            keys.append('fall_transition')

        data = {}
        if user_data is not None:
            for name in keys:
                cur_data = user_data[name]
                if isinstance(cur_data, Mapping):
                    val = user_data[name][sim_env_name]
                else:
                    val = cur_data
                data[name] = np.broadcast_to(val, delay_shape)
        elif zero_delay:
            for name in keys:
                data[name] = np.zeros(delay_shape)
        elif self.fake:
            for name in keys:
                val = 50.0e-12 if name.startswith('cell') else 20.0e-12
                data[name] = np.full(delay_shape, val)
        else:
            cur_specs = self._delay_specs.copy()
            cur_specs['in_pin'] = related
            cur_specs['out_pin'] = pin_name
            cur_specs['out_invert'] = out_invert
            cur_specs['out_rise'] = ttype.is_rising
            cur_specs['out_fall'] = ttype.is_falling
            if cond:
                pin_values = cur_specs['tbm_specs']['pin_values'].copy()
                pin_values.update(cond)
                update_recursive(cur_specs, pin_values, 'tbm_specs', 'pin_values')

            mm = sim_db.make_mm(CombLogicTimingMM, cur_specs)
            mm_result = await sim_db.async_simulate_mm_obj(f'{name}_{sim_id}', sim_dir / sim_id,
                                                           dut, mm)
            delay_data = mm_result.data['timing_data'][pin_name]

            for name in keys:
                # NOTE: remove corners
                data[name] = delay_data[name][0, ...]

        ans = dict(
            related=related,
            timing_type=ttype.name,
            cond=build_timing_cond_expr(cond),
            sense=sense_str,
            data=data,
        )
        output_list.append(ans)

    @staticmethod
    async def _measure_flop(name: str, sim_dir: Path, sim_db: SimulationDB,
                            dut: Optional[DesignInstance], seq_name: str, mm: MeasurementManager,
                            ans: Dict[str, Any]) -> None:
        sim_id = f'seq_timing_{seq_name}'
        result = await sim_db.async_simulate_mm_obj(f'{name}_{sim_id}', sim_dir / sim_id, dut, mm)
        for pin, timing_data in result.data.items():
            cur_info = ans[pin]
            timing_list = cur_info.get('timing', None)
            if timing_list is None:
                cur_info['timing'] = timing_data
            else:
                timing_list.extend(timing_data)

    async def _measure_custom(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                              dut: Optional[DesignInstance], meas_name: str, meas_cls: str,
                              meas_specs: Mapping[str, Any], ans: Dict[str, Any]) -> None:
        sim_env_name: str = self.specs['sim_env_name']
        mm_specs = dict(tbm_specs=self._tran_specs, fake=self.fake, sim_env_name=sim_env_name,
                        **meas_specs)
        mm = sim_db.make_mm(meas_cls, mm_specs)
        sim_id = f'custom_{meas_name}'
        mm_result = await sim_db.async_simulate_mm_obj(f'{name}_{sim_id}', sim_dir / sim_id,
                                                       dut, mm)
        for pin, timing_data in mm_result.data.items():
            cur_info = ans[pin]
            timing_list = cur_info.get('timing', None)
            if timing_list is None:
                cur_info['timing'] = timing_data
            else:
                timing_list.extend(timing_data)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

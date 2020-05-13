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

from typing import Any, Union, Tuple, Optional, Mapping, Sequence, List, Dict, cast

from pathlib import Path
from itertools import chain

import numpy as np

from bag.io.file import write_yaml
from bag.simulation.base import get_bit_list
from bag.simulation.data import SimData
from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from ..data.tran import EdgeType
from ..tran.digital import DigitalTranTB
from .util import setup_digital_tran


class CombLogicTimingMM(MeasurementManager):
    """Measures combinational logic delay and rise/fall time of a generic block.

    Assumes that no parameters/corners are swept.

    The results follow liberty file convention, i.e. delay_rise is the input-output delay
    for rising output edge.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : Union[str, Sequence[str]]
        input pin(s) to apply PWL waveform.
    out_pin : Union[str, Sequence[str]]
        output pin(s) to add capacitance loads.
    out_invert : Union[bool, Sequence[bool]]
        True if waveform at stop is inverted from start.  Corresponds to each start/stop pair.
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
            c_load :
                load capacitance.
            r_src :
                source resistance.  Only necessary if add_src_res is True.
    start_pin : Union[str, Sequence[str]]
        Defaults to in_pin.  Pins to start measuring delay.
    stop_pin : Union[str, Sequence[str]]
        Defaults to out_pin.  Pins to stop measuring delay.
    out_rise : bool
        Defaults to True.  True to return delay and rise/fall time for rising output edge.
    out_fall : bool
        Defaults to True.  True to return delay and rise/fall time for falling output edge.
    wait_cycles : int
        Defaults to 0.  Number of cycles to wait toggle before finally measuring delay.
    add_src_res : bool
        Defaults to False.  True to add source resistance.  Will use the variable "r_src".
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
    load_list : Optional[Sequence[Mapping[str, Any]]]
        Defaults to None.  Extra loads to add to testbench.
    fake: bool
        Defaults to False.  True to generate fake data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._in_list: Sequence[str] = []
        self._out_list: Sequence[str] = []
        self._start_list: Sequence[str] = []
        self._stop_list: Sequence[str] = []
        self._invert_list: Sequence[bool] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        in_pin: Union[str, Sequence[str]] = specs['in_pin']
        out_pin: Union[str, Sequence[str]] = specs['out_pin']
        out_invert: Union[bool, Sequence[bool]] = specs['out_invert']
        start_pin: Union[str, Sequence[str]] = specs.get('start_pin', '')
        stop_pin: Union[str, Sequence[str]] = specs.get('stop_pin', '')

        self._in_list = get_bit_list(in_pin)
        self._out_list = get_bit_list(out_pin)
        self._start_list = get_bit_list(start_pin) if start_pin else self._in_list
        self._stop_list = get_bit_list(stop_pin) if stop_pin else self._out_list

        if isinstance(out_invert, bool):
            self._invert_list = [out_invert] * len(self._start_list)
        else:
            self._invert_list = out_invert

        if len(self._stop_list) != len(self._start_list):
            raise ValueError('start/stop list length mismatch.')
        if len(self._invert_list) != len(self._start_list):
            raise ValueError('incorrect out_invert list length.')

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """A coroutine that performs measurement.

        The measurement is done like a FSM.  On each iteration, depending on the current
        state, it creates a new testbench (or reuse an existing one) and simulate it.
        It then post-process the simulation data to determine the next FSM state, or
        if the measurement is done.

        Parameters
        ----------
        name : str
            name of this measurement.
        sim_dir : Path
            simulation directory.
        sim_db : SimulationDB
            the simulation database object.
        dut : Optional[DesignInstance]
            the DUT to measure.

        Returns
        -------
        output : Dict[str, Any]
            the last dictionary returned by process_output().
        """
        specs = self.specs
        out_rise: bool = specs.get('out_rise', True)
        out_fall: bool = specs.get('out_fall', True)
        fake: bool = specs.get('fake', False)
        wait_cycles: int = specs.get('wait_cycles', 0)
        add_src_res: bool = specs.get('add_src_res', False)
        extra_loads: Optional[Sequence[Mapping[str, Any]]] = specs.get('load_list', None)

        rs = 'r_src' if add_src_res else ''
        load_list: List[Mapping[str, Any]] = [dict(pin=p_, type='cap', value='c_load')
                                              for p_ in self._out_list]
        if extra_loads:
            load_list.extend(extra_loads)
        pulse_list = [dict(pin=p_, tper='2*t_bit', tpw='t_bit', trf='t_rf',
                           td='t_bit', pos=True, rs=rs) for p_ in self._in_list]
        num_bits = 3 + 2 * wait_cycles

        tbm_specs, tb_params = setup_digital_tran(specs, dut, pulse_list=pulse_list,
                                                  load_list=load_list, skip_src=True)
        save_set = set(chain(self._in_list, self._out_list, self._start_list, self._stop_list))
        if add_src_res:
            save_set.update((DigitalTranTB.get_r_src_pin(p_) for p_ in self._in_list))
        tbm_specs['save_outputs'] = list(save_set)

        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))
        tbm.sim_params['t_sim'] = f't_rst+t_rst_rf/{tbm.trf_scale:.2f}+{num_bits}*t_bit'

        if fake:
            raise ValueError('fake mode is broken')
        else:
            results = await self._run_sim(name, sim_db, sim_dir, dut, tbm, tb_params, wait_cycles,
                                          out_rise, out_fall, is_mc=False)

        self.log(f'Measurement {name} done, recording results.')

        mc_params = specs.get('mc_params', {})
        if mc_params:
            mc_name = f'{name}_mc'
            self.log('Starting Monte Carlo simulation')
            mc_tbm_specs = tbm_specs.copy()
            mc_tbm_specs['sim_envs'] = [specs.get('mc_corner', 'tt_25')]
            mc_tbm_specs['monte_carlo_params'] = mc_params
            mc_tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, mc_tbm_specs))
            mc_tbm.sim_params['t_sim'] = f't_rst+t_rst_rf/{mc_tbm.trf_scale:.2f}+{num_bits}*t_bit'

            mc_results = await self._run_sim(mc_name, sim_db, sim_dir, dut, mc_tbm, tb_params,
                                             wait_cycles, out_rise, out_fall, is_mc=True)
            results = dict(
                tran=results,
                mc=mc_results,
            )

        write_yaml(sim_dir / f'{name}.yaml', results)
        return results

    async def _run_sim(self, name: str, sim_db: SimulationDB, sim_dir: Path, dut: DesignInstance,
                 tbm: DigitalTranTB, tb_params: Mapping[str, Any], wait_cycles: int,
                 out_rise: bool, out_fall: bool, is_mc: bool):
        sim_id = f'{name}_sim_delay'
        sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id,
                                                          dut, tbm, tb_params, tb_name=sim_id)
        data = sim_results.data

        trf_scale = tbm.trf_scale
        num_tbit = 1 + 2 * wait_cycles
        calc = tbm.get_calculator(data)
        t0 = calc.eval(f't_rst+{num_tbit}*t_bit+(t_rst_rf-t_rf)/{trf_scale:.2f}')
        timing_data = {}
        for start_pin, stop_pin, out_invert in zip(self._start_list, self._stop_list,
                                                   self._invert_list):
            pos_ipins, neg_ipins = tbm.get_diff_groups(start_pin)
            pos_opins, neg_opins = tbm.get_diff_groups(stop_pin)
            for opin in pos_opins:
                for ipin in pos_ipins:
                    timing_data[opin] = _get_timing_data(tbm, data, t0, opin, ipin,
                                                         out_invert, out_rise, out_fall, is_mc)
                for ipin in neg_ipins:
                    timing_data[opin] = _get_timing_data(tbm, data, t0, opin, ipin,
                                                         not out_invert, out_rise, out_fall, is_mc)
            for opin in neg_opins:
                for ipin in pos_ipins:
                    timing_data[opin] = _get_timing_data(tbm, data, t0, opin, ipin,
                                                         not out_invert, out_fall, out_rise, is_mc)
                for ipin in neg_ipins:
                    timing_data[opin] = _get_timing_data(tbm, data, t0, opin, ipin,
                                                         out_invert, out_fall, out_rise, is_mc)

        results = dict(
            sim_envs=tbm.sim_envs,
            sim_params=tbm.get_calculator(data).namespace,
            timing_data=timing_data,
        )
        return results

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')


def _get_timing_data(tbm: DigitalTranTB, data: SimData, t0: np.ndarray,
                     out_pin: str, in_pin: str, out_invert: bool, out_rise: bool, out_fall: bool,
                     is_mc: bool) -> Mapping[str, Any]:
    cur_data = dict(related=in_pin)
    in_edge = EdgeType.FALL if out_invert else EdgeType.RISE
    if out_rise:
        tdr = tbm.calc_delay(data, in_pin, out_pin, in_edge, EdgeType.RISE, t_start=t0)
        tr = tbm.calc_trf(data, out_pin, True, t_start=t0)
        if is_mc:
            cur_data['cell_rise_std'] = np.std(tdr, axis=-1)
            cur_data['rise_transition_std'] = np.std(tr, axis=-1)
        else:
            cur_data['cell_rise'] = tdr
            cur_data['rise_transition'] = tr
    if out_fall:
        tdf = tbm.calc_delay(data, in_pin, out_pin, in_edge.opposite, EdgeType.FALL,
                             t_start=t0)
        tf = tbm.calc_trf(data, out_pin, False, t_start=t0)
        if is_mc:
            cur_data['fall_rise_std'] = np.std(tdf, axis=-1)
            cur_data['fall_transition_std'] = np.std(tf, axis=-1)
        else:
            cur_data['cell_fall'] = tdf
            cur_data['fall_transition'] = tf
    return cur_data

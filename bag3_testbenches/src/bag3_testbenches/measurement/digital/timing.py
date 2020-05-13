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

from typing import Dict, Any, Union, Sequence, Tuple, Optional, List, Mapping, Callable, Type

from pathlib import Path

import numpy as np

from pybag.core import FileLogger

from bag.io import open_file
from bag.design.module import Module
from bag.simulation.data import AnalysisType, SimData, SimNetlistInfo, netlist_info_from_dict
from bag.simulation.core import TestbenchManager

from ...schematic.digital_tb_tran import bag3_testbenches__digital_tb_tran
from ..data.tran import bits_to_pwl_iter, get_first_crossings, EdgeType


class CombLogicTimingTB(TestbenchManager):
    """This class performs timing measurements on combinational logics.

    Assumptions:

    1. tbit is not swept.
    2. power supplies are not swept.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, Any]
        Required entries are listed below.  In addition, all input output power domain should
        have their voltage as parameters here.

        tbit : float
            the input data waveform bit period.
        trf : float
            input data rise/fall time, as measured with thres_lo/thres_hi.

        Finally, CombLogicTimingTB will define a tsim parameter that is a function of tbit.

    thres_lo : float
        low threshold for rise/fall time calculation, as fraction of VDD.
    thres_hi : float
        high threshold for rise/fall time calculation, as fraction of VDD.
    stimuli_pwr : str
        stimuli voltage power domain parameter name.
    save_outputs : Sequence[str]
        list of nets to save in simulation data file.
    rtol : float
        relative tolerance for equality checking in timing measurement.
    atol : float
        absolute tolerance for equality checking in timing measurement.
    tran_options : Mapping[str, Any]
        transient simulation options dictionary.

    nbit_delay : int
        Optional.  Delay in number of bits.  Defaults to 0.
    gen_invert : bool
        Optional.  True to generate complementary input waveform on net "inbar".  Defaults to False.
    ctrl_params : Mapping[str, Sequence[str]]
        Optional.  If given, will simulation multiple input pulses, changing control signal values
        between each pulse.  The keys are control signal net names, and values are a list of
        control signal values for each input pulse.  The length of the values list must be the same
        for all control signals.
    clk_params : Dict[str, Any]
        Optional.  If specified, generate a clock waveform.  It has the following entries:

        thres_lo : float
            low threshold for rise/fall time definition.
        thres_hi : float
            high threshold for rise/fall time definition.
        trf : Union[float, str]
            clock rise/fall time, either number in seconds or variable name.
        tper : Union[float, str]
            clock period, either number in seconds or variable name.
        nper : int
            number of clock periods to generate.
        clk_delay : Union[float, str]
            the clock delay, either number in seconds or variable name.
        clk_pwr : str
            the clock power domain parameter name.  Defaults to 'vdd'.

    clk_invert: bool
        Optional.  True to generate inverted clock waveform on net "clkb".  Defaults to False.
    tstep : Optional[float]
        Optional.  The strobe period.  Defaults to no strobing.
    write_numbers : bool
        Optional.  True to write numbers in generated PWL files.  Defaults to False.

    print_delay_list : List[Union[Tuple[str, str], Tuple[str, str, str, str]]]
        list of delays to print out in summary report.
    print_trf_list : List[Union[str, Tuple[str, str]]]
        list of rise/fall times to print out in summary report.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__digital_tb_tran

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        """Set up PWL waveform files."""
        if sch_params is None:
            return sch_params

        specs = self.specs
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        write_numbers: bool = specs.get('write_numbers', False)
        in_pwr: str = specs.get('stimuli_pwr', 'vdd')
        gen_invert: bool = specs.get('gen_invert', False)
        clk_invert: bool = specs.get('clk_invert', False)
        clk_params: Mapping[str, Any] = specs.get('clk_params', {})
        ctrl_params: Mapping[str, Sequence[str]] = specs.get('ctrl_params', {})

        sim_params = self.sim_params
        in_pwr_val = sim_params[in_pwr]

        sch_in_files: Optional[List[Tuple[str, str]]] = sch_params.get('in_file_list', None)
        sch_clk_files: Optional[List[Tuple[str, str]]] = sch_params.get('clk_file_list', None)

        swp_info = self.swp_info
        local_write_numbers = True
        if swp_info:
            if isinstance(swp_info, Mapping):
                if 'tbit' in swp_info or 'trf' in swp_info or in_pwr in swp_info:
                    local_write_numbers = False
            else:
                for l in swp_info:
                    if 'tbit' in l or 'trf' in l or in_pwr in l:
                        local_write_numbers = False

        write_numbers = write_numbers or local_write_numbers

        in_file_list = []
        clk_file_list = []
        if sch_in_files:
            in_file_list.extend(sch_in_files)
        if sch_clk_files:
            clk_file_list.extend(sch_clk_files)

        nbit_delay, num_runs = self._calc_sim_time_info()
        trf_scale = thres_hi - thres_lo
        if ctrl_params:
            for ctrl_sig_name, ctrl_sig_vals in ctrl_params.items():
                cur_len = len(ctrl_sig_vals)
                if cur_len != num_runs:
                    self.error(f'control signal {ctrl_sig_name} values '
                               f'length = {cur_len} != {num_runs}')
                sig_path = self.work_dir / f'{ctrl_sig_name}_pwl.txt'
                _write_pwl_file(sig_path, ctrl_sig_vals, sim_params, 'tbit', 'trf', trf_scale,
                                nbit_delay, write_numbers, tb_scale=3)
                in_file_list.append((ctrl_sig_name, str(sig_path.resolve())))

        # generate PWL waveform files
        in_data = [0, in_pwr_val, 0] * num_runs
        in_path = self.work_dir / 'in_pwl.txt'
        _write_pwl_file(in_path, in_data, sim_params, 'tbit', 'trf', trf_scale, nbit_delay,
                        write_numbers)
        in_file_list.append(('in', str(in_path.resolve())))

        if gen_invert:
            in_data_bar = [in_pwr_val, 0, in_pwr_val] * num_runs
            inbar_path = self.work_dir / 'inbar_pwl.txt'
            _write_pwl_file(inbar_path, in_data_bar, sim_params, 'tbit', 'trf', trf_scale,
                            nbit_delay, write_numbers)
            in_file_list.append(('inbar', str(inbar_path.resolve())))

        if clk_params:
            clk_thres_hi: float = clk_params['thres_hi']
            clk_thres_lo: float = clk_params['thres_lo']
            clk_trf: Union[float, str] = clk_params['trf']
            clk_tper: Union[float, str] = clk_params['tper']
            clk_nper: int = clk_params.get('nper', 3)
            clk_delay: Union[float, str] = clk_params.get('clk_delay', '')
            clk_pwr: str = clk_params.get('clk_pwr', 'vdd')

            clk_pwr_val = sim_params[clk_pwr]
            clk_data = [0, clk_pwr_val] * clk_nper
            clk_data.append(0)
            clk_trf_scale = clk_thres_hi - clk_thres_lo
            clk_path = self.work_dir / 'in_clk.txt'
            _write_pwl_file(clk_path, clk_data, sim_params, clk_tper, clk_trf, clk_trf_scale,
                            0, write_numbers, tb_scale=0.5, delay=clk_delay)
            clk_file_list.append(('clk', str(clk_path.resolve())))

            if clk_invert:
                clkb_data = ['vdd', '0'] * clk_nper
                clkb_data.append('vdd')
                clkb_path = self.work_dir / 'inb_clk.txt'
                _write_pwl_file(clkb_path, clkb_data, sim_params, clk_tper, clk_trf, clk_trf_scale,
                                0, write_numbers, tb_scale=0.5, delay=clk_delay)
                clk_file_list.append(('clkb', str(clkb_path.resolve())))

        ans = {k: v for k, v in sch_params.items()}
        ans['in_file_list'] = in_file_list
        ans['clk_file_list'] = clk_file_list
        return ans

    def get_netlist_info(self) -> SimNetlistInfo:
        # define tsim parameter
        nbit_delay, num_runs = self._calc_sim_time_info()
        num_tbit = nbit_delay + 3 * num_runs
        self.sim_params['tsim'] = f'{num_tbit}*tbit'

        specs = self.specs
        tstep: Optional[float] = specs.get('tstep', None)
        tran_options: Mapping[str, Any] = specs.get('tran_options', {})
        save_outputs: Sequence[str] = specs.get('save_outputs', [])

        tran_dict = dict(type='TRAN',
                         start=0.0,
                         stop='tsim',
                         options=tran_options,
                         save_outputs=save_outputs,
                         )
        if tstep is not None:
            tran_dict['strobe'] = tstep

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [tran_dict]
        return netlist_info_from_dict(sim_setup)

    def print_results(self, data: SimData) -> None:
        """Override to print results."""
        specs = self.specs
        delay_list: List[Tuple[Any, ...]] = specs.get('print_delay_list', [])
        trf_list: List[Union[str, Tuple[str, str]]] = specs.get('print_trf_list', [])

        for ele in delay_list:
            if len(ele) == 3:
                out_invert, in_name, out_name = ele
                in_pwr = out_pwr = 'vdd'
            elif len(ele) == 5:
                out_invert, in_name, out_name, in_pwr, out_pwr = ele
            else:
                out_invert = False
                in_name = out_name = in_pwr = out_pwr = ''
                self.error(f'Unknown print_delay element: {ele}')

            tdr, tdf = self.calc_output_delay(data, in_name, out_name, out_invert,
                                              in_pwr=in_pwr, out_pwr=out_pwr)
            self.log(f'in_rise {in_name}/{out_name} td(s):\n{tdr}')
            self.log(f'in_fall {in_name}/{out_name} td(s):\n{tdf}')

        for ele in trf_list:
            if isinstance(ele, str):
                out_name = ele
                out_pwr = 'vdd'
            else:
                out_name, out_pwr = ele
            tr, tf = self.calc_output_trf(data, out_name, out_pwr=out_pwr)
            self.log(f'{out_name} tr (s):\n{tr}')
            self.log(f'{out_name} tf (s):\n{tf}')

    def calc_output_delay(self, data: SimData, in_name: str, out_name: str, out_invert: bool,
                          shape: Optional[Tuple[int, ...]] = None, in_pwr: str = 'vdd',
                          out_pwr: str = 'vdd') -> Tuple[np.ndarray, np.ndarray]:
        return self.get_output_delay(data, self.specs, in_name, out_name, out_invert,
                                     shape=shape, in_pwr=in_pwr, out_pwr=out_pwr)

    def calc_output_trf(self, data: SimData, out_name: str,
                        shape: Optional[Tuple[int, ...]] = None, out_pwr: str = 'vdd',
                        allow_inf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_output_trf(data, self.specs, out_name, shape=shape, out_pwr=out_pwr,
                                   allow_inf=allow_inf, logger=self.logger)

    def _calc_sim_time_info(self) -> Tuple[int, int]:
        return self._get_sim_time_info(self.specs)

    @classmethod
    def get_output_delay(cls, data: SimData, specs: Mapping[str, Any], in_name: str, out_name: str,
                         out_invert: bool, shape: Optional[Tuple[int, ...]] = None,
                         in_pwr: str = 'vdd', out_pwr: str = 'vdd'
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute delay from simulation data.

        if the output never resolved correctly, infinity is returned.

        Parameters
        ----------
        data : SimData
            Simulation data.
        specs : Dict[str, Any]
            testbench specs.
        in_name : str
            input signal name.
        out_name : str
            output signal name.
        out_invert : bool
            True if output is inverted from input.
        shape : Optional[Tuple[int, ...]]
            the delay result output shape.
        in_pwr : str
            input supply voltage variable name.
        out_pwr : str
            output supply voltage variable name.

        Returns
        -------
        tdr : np.ndarray
            array of output delay for rising input edge.
        tdf : np.ndarray
            array of output delay for falling input edge.
        """
        # TODO: remove
        sim_params: Mapping[str, float] = specs['sim_params']
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        nbit_delay, num_runs = cls._get_sim_time_info(specs)

        tbit = sim_params['tbit']
        vdd_in = sim_params[in_pwr]
        vdd_out = sim_params[out_pwr]

        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        in_vec = data[in_name]
        out_vec = data[out_name]

        vth_in = vdd_in / 2
        vth_out = vdd_out / 2
        t0 = nbit_delay * tbit
        trun = 3 * tbit
        tdr_list = []
        tdf_list = []
        for run_idx in range(num_runs):
            start = t0 + run_idx * trun
            tdr, tdf = cls.compute_output_delay(tvec, in_vec, out_vec, vth_in, vth_out,
                                                out_invert, start=start, stop=start + trun,
                                                rtol=rtol, atol=atol, shape=shape)
            tdr_list.append(tdr)
            tdf_list.append(tdf)

        if num_runs == 1:
            return tdr_list[0], tdf_list[0]
        return np.array(tdr_list), np.array(tdf_list)

    @classmethod
    def get_output_trf(cls, data: SimData, specs: Mapping[str, Any], out_name: str,
                       shape: Optional[Tuple[int, ...]] = None, out_pwr: str = 'vdd',
                       allow_inf: bool = False, logger: Optional[FileLogger] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute output rise/fall time from simulation data.

        if output never crosses the high threshold, infinity is returned.  If output never
        crosses the low threshold, nan is returned.

        Parameters
        ----------
        data : SimData
            Simulation data.
        specs : Dict[str, Any]
            testbench specs.
        out_name : str
            output signal name.
        shape : Optional[Tuple[int, ...]]
            the delay result output shape.
        out_pwr : str
            output supply voltage variable name.
        allow_inf: bool
            Turns off error checking for infinity values
            Useful for really slow rise/fall times and/or large Cload where the transition is
            not complete
        logger : Optional[FileLogger]
            the optional logger object.

        Returns
        -------
        tdr : np.ndarray
            array of output delay for rising input edge.
        tdf : np.ndarray
            array of output delay for falling input edge.
        """
        # TODO: remove
        sim_params: Mapping[str, float] = specs['sim_params']
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)

        nbit_delay, num_runs = cls._get_sim_time_info(specs)

        vdd_out = sim_params[out_pwr]
        tbit = sim_params['tbit']

        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data[out_name]

        vlo = vdd_out * thres_lo
        vhi = vdd_out * thres_hi

        t0 = nbit_delay * tbit
        trun = 3 * tbit
        tr_list = []
        tf_list = []
        for run_idx in range(num_runs):
            start = t0 + run_idx * trun
            tr, tf = cls.compute_output_trf(tvec, yvec, vlo, vhi, start=start, stop=start + trun,
                                            rtol=rtol, atol=atol, shape=shape, allow_inf=allow_inf,
                                            logger=logger)
            tr_list.append(tr)
            tf_list.append(tf)

        if num_runs == 1:
            return tr_list[0], tf_list[0]
        return np.array(tr_list), np.array(tf_list)

    @classmethod
    def compute_output_delay(cls, tvec: np.ndarray, in_vec: np.ndarray, out_vec: np.ndarray,
                             vth_in: float, vth_out: float, out_invert: bool, **kwargs
                             ) -> Tuple[np.ndarray, np.ndarray]:
        in_r = get_first_crossings(tvec, in_vec, vth_in, etype=EdgeType.RISE, **kwargs)
        in_f = get_first_crossings(tvec, in_vec, vth_in, etype=EdgeType.FALL, **kwargs)
        out_r = get_first_crossings(tvec, out_vec, vth_out, etype=EdgeType.RISE, **kwargs)
        out_f = get_first_crossings(tvec, out_vec, vth_out, etype=EdgeType.FALL, **kwargs)

        if out_invert:
            out_f -= in_r
            out_r -= in_f
            return out_f, out_r
        else:
            out_r -= in_r
            out_f -= in_f
            return out_r, out_f

    @classmethod
    def compute_output_trf(cls, tvec: np.ndarray, yvec: np.ndarray, vlo: float, vhi: float,
                           allow_inf: bool = False, logger: Optional[FileLogger] = None,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        print_fun = print if logger is None else logger.warn
        tr0 = get_first_crossings(tvec, yvec, vlo, etype=EdgeType.RISE, **kwargs)
        tr1 = get_first_crossings(tvec, yvec, vhi, etype=EdgeType.RISE, **kwargs)
        tf0 = get_first_crossings(tvec, yvec, vhi, etype=EdgeType.FALL, **kwargs)
        tf1 = get_first_crossings(tvec, yvec, vlo, etype=EdgeType.FALL, **kwargs)
        tr = cls._trf_helper(tr0, tr1, allow_inf, 'trise', print_fun)
        tf = cls._trf_helper(tf0, tf1, allow_inf, 'tfall', print_fun)
        return tr, tf

    @classmethod
    def _trf_helper(cls, t0: np.ndarray, t1: np.ndarray, allow_inf: bool, tag: str,
                    print_fun: Callable[[str], None]) -> np.ndarray:
        has_nan = np.isnan(t0).any() or np.isnan(t1).any()
        has_inf = np.isinf(t0).any() or np.isinf(t1).any()
        if has_nan or (has_inf and not allow_inf):
            print_fun(f'Got an invalid value in computing {tag}, values:')
            print_fun(str(t0))
            print_fun(str(t1))
            t1 = np.full(t1.shape, np.inf)
        else:
            t1 -= t0

        return t1

    @classmethod
    def _get_sim_time_info(cls, specs: Mapping[str, Any]) -> Tuple[int, int]:
        # TODO: remove
        nbit_delay: int = specs.get('nbit_delay', 0)
        ctrl_params: Optional[Mapping[str, Sequence[str]]] = specs.get('ctrl_params', None)
        num_runs = len(next(iter(ctrl_params.values()))) if ctrl_params is not None else 1
        return nbit_delay, num_runs


def _write_pwl_file(path: Path, data: Union[Sequence[str], Sequence[float]],
                    sim_params: Mapping[str, Union[float, str]],
                    tbit: Union[float, str], trf: Union[float, str], trf_scale: float,
                    nbit_delay: int, write_numbers: bool, tb_scale: float = 1,
                    delay: Union[float, str] = '') -> None:
    with open_file(path, 'w') as f:
        if write_numbers:
            tbit_val = sim_params[tbit] if isinstance(tbit, str) else tbit
            trf_val = sim_params[trf] if isinstance(trf, str) else trf
            if isinstance(delay, str):
                delay_val = sim_params[delay] if delay else 0
            else:
                delay_val = delay
            if isinstance(tbit_val, str) or isinstance(trf_val, str) or isinstance(delay_val, str):
                raise ValueError('Cannot write numeric pwl file.')
            for _, s_tb, s_tr, val in bits_to_pwl_iter(data):
                time_val = (delay_val + tbit_val * (tb_scale * s_tb + nbit_delay) +
                            trf_val * s_tr / trf_scale)
                f.write(f'{time_val} {val}\n')
        else:
            prefix = delay + '+' if delay else ''
            for _, s_tb, s_tr, val in bits_to_pwl_iter(data):
                f.write(f'{prefix}{tbit}*{tb_scale * s_tb + nbit_delay:.4g}+'
                        f'{trf}*({s_tr / trf_scale:.4g}) {val}\n')

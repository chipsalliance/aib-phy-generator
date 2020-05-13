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

from typing import Any, Sequence, Tuple, Set, Mapping, Dict, Iterable

from pybag.core import get_cdba_name_bits

from bag3_liberty.data import parse_cdba_name

from ...data.tran import EdgeType
from .base import FlopTimingBase, FlopMeasMode, FlopInputMode


class FlopArrayTimingTB(FlopTimingBase):
    """This class performs transient simulation on an array of flops

    this testbench assumes the following:

    1. all flops share the same clock, reset, and scan enable signal.
    2. output of one flop goes into the scan_in of the next one.

    Notes
    -----
    specification dictionary has the following entries in addition to those in FlopTimingBase:

    flop_params : Mapping[str, Any]
        Flop parameters, with the following entries:

        in_pin : str
            the input pin(s), in CDBA format.  the first pin in the string corresponds to the
            first flop in the scan chain.
        out_pin : str
            the output pin(s), in CDBA format.
        clk_pin : str
            the clock pin name.
        se_pin : str
            Optional.  The scan enable pin name.
        si_pin : str
            Optional.  The first scan input pin.
        rst_pin : str
            Optional.  The reset pin name.
        rst_active_high : bool
            Defaults to True.  True if reset pin is active high.
        rst_to_high : bool
            Defaults to False.  True if output is high during reset.
        rst_timing : bool
            True to characterize timing on reset pin.
        out_invert : bool
            Defaults to False.  True if outputs are inverted from input.
        clk_rising : bool
            True if flop trigger on rising edge of clock.
        out_timing_pin : str
            Defaults to out_pin.  The output pin(s) for which to add clock-to-q delay information,
            as a string in CDBA format.
        in_timing_pin: str
            Defaults to in_pin.  The input pin(s) for which to add timing information,
            as a string in CDBA format.
        c_load_pin: str
            Defaults to out_timing_pin.  The output pin(s) for which to add load capacitor.
        setup_offset : float
            Defaults to 0.  Offset to add to setup time.
        hold_offset : float
            Defaults to 0.  Offset to add to hold time.
        delay_offset : float
            Defaults to 0.  Offset to add to clock-to-q delay.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._in_list: Sequence[str] = []
        self._out_list: Sequence[str] = []
        self._in_timing_set: Set[str] = set()
        self._out_timing_set: Set[str] = set()
        self._c_load_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        # NOTE: initialize in_list and out_list before running super's initialization,
        # so get_stimuli() behaves correctly.
        flop_params: Mapping[str, Any] = self.specs['flop_params']
        in_pin: str = flop_params['in_pin']
        out_pin: str = flop_params['out_pin']
        si_pin: str = flop_params.get('si_pin', '')
        in_timing_pin: str = flop_params.get('in_timing_pin', in_pin)
        out_timing_pin: str = flop_params.get('out_timing_pin', out_pin)
        c_load_pin: str = flop_params.get('out_timing_pin', out_timing_pin)

        self._in_list = get_cdba_name_bits(in_pin)
        self._out_list = get_cdba_name_bits(out_pin)
        if len(self._in_list) != len(self._out_list):
            raise ValueError('different number of input and output pins')

        if in_timing_pin:
            self._in_timing_set = set(get_cdba_name_bits(in_timing_pin))
        else:
            self._in_timing_set = set()
        if si_pin:
            self._in_timing_set.add(si_pin)

        if out_timing_pin:
            self._out_timing_set = set(get_cdba_name_bits(out_timing_pin))
        else:
            self._out_timing_set = set()
        if c_load_pin:
            self._c_load_list = get_cdba_name_bits(c_load_pin)
        else:
            self._c_load_list = []

        super().commit()

    @property
    def num_cycles(self) -> int:
        if self.meas_mode.is_reset:
            return 1
        return 2

    @property
    def c_load_pins(self) -> Iterable[str]:
        return self._c_load_list

    @classmethod
    def get_default_flop_params(cls) -> Dict[str, Any]:
        return dict(se_pin='', si_pin='', rst_pin='', rst_active_high=True,
                    rst_to_high=False, out_invert=False, clk_rising=True,
                    setup_offset=0, hold_offset=0, delay_offset=0)

    @classmethod
    def get_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        se_pin: str = flop_params.get('se_pin', '')
        si_pin: str = flop_params.get('si_pin', '')
        rst_pin: str = flop_params.get('rst_pin', '')
        rst_timing: bool = flop_params.get('rst_timing', True)
        clk_rising: bool = flop_params.get('clk_rising', True)
        in_timing_pin: str = flop_params.get('in_timing_pin', flop_params['in_pin'])

        ans = []
        if in_timing_pin:
            ans.append(FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))
        if se_pin and si_pin:
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SE, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SE, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SE, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SE, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))

            ans.append(FlopMeasMode(in_mode=FlopInputMode.SI, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SI, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SI, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.SI, in_rising=False,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=False))
        if rst_pin and rst_timing:
            ans.append(FlopMeasMode(in_mode=FlopInputMode.RECOVERY, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
            ans.append(FlopMeasMode(in_mode=FlopInputMode.REMOVAL, in_rising=True,
                                    setup_rising=clk_rising, hold_rising=clk_rising,
                                    meas_setup=True))
        return ans

    @classmethod
    def get_output_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        clk_rising: bool = flop_params.get('clk_rising', True)
        ans = [FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=clk_rising,
                            hold_rising=clk_rising, meas_setup=True),
               FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=clk_rising,
                            hold_rising=clk_rising, meas_setup=True),
               ]
        return ans

    @classmethod
    def get_setup_hold_name(cls, pin: str) -> Tuple[str, str]:
        basename, bus_range = parse_cdba_name(pin)
        if bus_range is None:
            var_setup = f't_setup_{basename}_'
            var_hold = f't_hold_{basename}_'
        else:
            var_setup = f't_setup_{basename}_{bus_range[0]}'
            var_hold = f't_hold_{basename}_{bus_range[0]}'
        return var_setup, var_hold

    @classmethod
    def get_recovery_removal_name(cls, pin: str) -> Tuple[str, str]:
        basename, bus_range = parse_cdba_name(pin)
        if bus_range is None:
            v1 = f't_recovery_{basename}_'
            v2 = f't_removal_{basename}_'
        else:
            v1 = f't_recovery_{basename}_{bus_range[0]}'
            v2 = f't_removal_{basename}_{bus_range[0]}'
        return v1, v2

    def get_stimuli(self) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, int], Set[str],
                                   Sequence[str]]:
        mode = self.meas_mode
        flop_params = self.flop_params

        clk_pin: str = flop_params['clk_pin']
        se_pin: str = flop_params['se_pin']
        si_pin: str = flop_params['si_pin']
        rst_pin: str = flop_params['rst_pin']
        rst_active_high: bool = flop_params['rst_active_high']
        rst_to_high: bool = flop_params['rst_to_high']
        clk_rising: bool = flop_params['clk_rising']
        if clk_rising != mode.is_pos_edge_clk:
            raise ValueError(f'Cannot perform measurement {mode.name}, clk phase is wrong.')

        pulses = [self.get_clk_pulse(clk_pin, clk_rising)]
        biases = {}
        var_list = []
        if mode.is_input or mode.is_scan_in:
            pos = mode.input_rising
            if mode.is_scan_in:
                if not se_pin or not si_pin:
                    raise ValueError('No scan pins defined.')

                stimuli_pins = [si_pin]
                for pin in self._in_list:
                    biases[pin] = 0
                biases[se_pin] = 1
            else:
                stimuli_pins = self._in_list
                if se_pin and si_pin:
                    biases[si_pin] = biases[se_pin] = 0

            if rst_pin:
                pulses.append(self.get_rst_pulse(rst_pin, rst_active_high))

            for pin in stimuli_pins:
                var_setup, var_hold = self.get_setup_hold_name(pin)
                var_list.append(var_setup)
                var_list.append(var_hold)
                pulses.append(self.get_input_pulse(pin, var_setup, var_hold, pos))
        elif mode.is_reset:
            if not rst_pin:
                raise ValueError('No reset pin defined.')

            if se_pin and si_pin:
                biases[si_pin] = biases[se_pin] = 0

            in_val = int(not rst_to_high)
            for pin in self._in_list:
                biases[pin] = in_val

            is_recovery = mode.is_recovery
            var_name = self.get_recovery_removal_name(rst_pin)[int(not is_recovery)]
            var_list.append(var_name)
            pulses.append(self.get_rst_pulse(rst_pin, rst_active_high, var_name=var_name,
                                             is_recovery=is_recovery))
        elif mode.is_scan_en:
            if not se_pin or not si_pin:
                raise ValueError('No scan pins defined.')

            if rst_pin:
                pulses.append(self.get_rst_pulse(rst_pin, rst_active_high))

            var_setup, var_hold = self.get_setup_hold_name(se_pin)
            clk_mid = 't_clk_delay+t_clk_per'
            var_list.append(var_setup)
            var_list.append(var_hold)

            if mode.is_scan_en and mode.input_rising:
                for idx, pin in enumerate(self._in_list):
                    biases[pin] = (idx & 1)
                biases[si_pin] = 1
                pulses.append(self.get_input_pulse(se_pin, var_setup, var_hold, True))
            else:
                biases[si_pin] = 0
                for pin in self._in_list:
                    pulses.append(dict(pin=pin, tper='2*t_sim', tpw='t_sim',
                                       trf='t_clk_rf', td='t_clk_delay+t_clk_per/2', pos=True))

                pulses.append(dict(pin=se_pin, tper=f't_clk_per/2+{var_hold}',
                                   tpw=f't_clk_per/2-{var_setup}', trf='t_rf',
                                   td=f'{clk_mid}-t_clk_per/2', pos=True))
        else:
            raise ValueError(f'Unsupported flop measurement mode: {mode}')

        outputs = set(self._in_list)
        outputs.update(self._out_list)
        outputs.add(clk_pin)
        if se_pin:
            outputs.add(se_pin)
        if si_pin:
            outputs.add(si_pin)
        if rst_pin:
            outputs.add(rst_pin)
        return pulses, biases, outputs, var_list

    def get_output_map(self, output_timing: bool
                       ) -> Mapping[str, Tuple[Mapping[str, Any],
                                               Sequence[Tuple[EdgeType, Sequence[str]]]]]:
        mode = self.meas_mode
        flop_params = self.flop_params
        clk_pin: str = flop_params['clk_pin']
        se_pin: str = flop_params['se_pin']
        rst_pin: str = flop_params['rst_pin']
        out_invert: bool = flop_params['out_invert']
        rst_to_high: bool = flop_params['rst_to_high']
        rst_active_high: bool = flop_params['rst_active_high']
        setup_offset: float = flop_params['setup_offset']
        hold_offset: float = flop_params['hold_offset']
        delay_offset: float = flop_params['delay_offset']

        meas_setup = mode.meas_setup
        sh_idx = int(not meas_setup)

        if output_timing:
            offset = delay_offset
        elif mode.is_recovery or meas_setup:
            offset = setup_offset
        else:
            offset = hold_offset

        if rst_pin:
            if rst_active_high:
                cond_rst_off = f'!{rst_pin}'
            else:
                cond_rst_off = rst_pin
        else:
            cond_rst_off = ''

        if se_pin and not output_timing:
            cond_se_off = f'!{se_pin}'
        else:
            cond_se_off = ''

        ans = {}
        if mode.is_input or mode.is_scan_in:
            if mode.is_input:
                in_out_iter = zip(self._in_list, self._out_list)
                cond_str = ' && '.join((val for val in [cond_se_off, cond_rst_off] if val))
            else:
                si_pin: str = flop_params['si_pin']
                if not si_pin or not se_pin:
                    raise ValueError('No scan in pin defined.')
                in_out_iter = [(si_pin, self._out_list[0])]
                cond_str = se_pin if not rst_pin else f'{se_pin} && {cond_rst_off}'

            for in_pin, out_pin in in_out_iter:
                if ((output_timing and out_pin in self._out_timing_set) or
                        (not output_timing and in_pin in self._in_timing_set)):
                    var_name = self.get_setup_hold_name(in_pin)[sh_idx]
                    out_diff_grp = self.get_diff_groups(out_pin)
                    rise_idx = int(not (mode.input_rising ^ out_invert))
                    timing_info = self.get_timing_info(mode, [in_pin], clk_pin, cond_str,
                                                       rst_active_high, offset=offset)
                    edge_out_list = [(EdgeType.RISE, out_diff_grp[rise_idx]),
                                     (EdgeType.FALL, out_diff_grp[rise_idx ^ 1])]
                    ans[var_name] = (timing_info, edge_out_list)
        elif mode.is_scan_en:
            if not se_pin:
                raise ValueError('No scan en pin defined.')

            cond_str = se_pin if not rst_pin else f'{se_pin} && {cond_rst_off}'
            rf_list = ([], [])
            for out_idx, out_pin in enumerate(self._out_list):
                out_diff_grp = self.get_diff_groups(out_pin)
                out_falling = mode.is_scan_en and mode.input_rising and (out_idx & 1)
                pos_rf_idx = int(out_invert ^ out_falling)
                rf_list[pos_rf_idx].extend(out_diff_grp[0])
                rf_list[pos_rf_idx ^ 1].extend(out_diff_grp[1])

            var_name = self.get_setup_hold_name(se_pin)[sh_idx]
            timing_info = self.get_timing_info(mode, [se_pin], clk_pin, cond_str, rst_active_high,
                                               offset=offset)
            edge_out_list = [(EdgeType.RISE, rf_list[0]), (EdgeType.FALL, rf_list[1])]
            ans[var_name] = (timing_info, edge_out_list)
        elif mode.is_reset:
            if not rst_pin:
                raise ValueError('No reset pin defined.')

            out_falling = not ((not rst_to_high) ^ out_invert)
            rf_list = ([], [])
            for out_pin in self._out_list:
                out_diff_grp = self.get_diff_groups(out_pin)
                pos_rf_idx = int(out_invert ^ out_falling)
                rf_list[pos_rf_idx].extend(out_diff_grp[0])
                rf_list[pos_rf_idx ^ 1].extend(out_diff_grp[1])

            var_re, var_rm = self.get_recovery_removal_name(rst_pin)
            var_name = var_re if mode.is_recovery else var_rm
            timing_info = self.get_timing_info(mode, [rst_pin], clk_pin, '', rst_active_high,
                                               offset=offset)
            edge_out_list = [(EdgeType.RISE, rf_list[0]), (EdgeType.FALL, rf_list[1])]
            ans[var_name] = (timing_info, edge_out_list)
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        return ans

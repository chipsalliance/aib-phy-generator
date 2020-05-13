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

from typing import Dict, Any, List, Tuple, Optional, Iterable, Mapping, Sequence

import asyncio
from pathlib import Path
from itertools import chain

from pybag.enum import LogLevel

from bag.io.file import read_yaml
from bag.simulation.base import get_corner_temp
from bag.core import BagProject

from bag3_liberty.enum import LogicType, TermType, LUTType
from bag3_liberty.data import Library, Cell, parse_cdba_name, get_bus_bit_name

from .char import LibertyCharMM


def generate_liberty(prj: BagProject, lib_config: Mapping[str, Any],
                     sim_config: Mapping[str, Any], specs: Mapping[str, Any],
                     fake: bool = False, extract: bool = False,
                     force_sim: bool = False, force_extract: bool = False,
                     gen_all_env: bool = False, gen_sch: bool = False, export_lay: bool = False,
                     log_level: LogLevel = LogLevel.DEBUG) -> None:
    asyncio.run(async_generate_liberty(prj, lib_config, sim_config, specs, fake=fake,
                                       extract=extract, force_sim=force_sim,
                                       force_extract=force_extract, gen_sch=gen_sch,
                                       gen_all_env=gen_all_env, export_lay=export_lay,
                                       log_level=log_level))


async def async_generate_liberty(prj: BagProject, lib_config: Mapping[str, Any],
                                 sim_config: Mapping[str, Any], cell_specs: Mapping[str, Any],
                                 fake: bool = False, extract: bool = False,
                                 force_sim: bool = False, force_extract: bool = False,
                                 gen_all_env: bool = False, gen_sch: bool = False,
                                 export_lay: bool = False, log_level: LogLevel = LogLevel.DEBUG
                                 ) -> None:
    """Generate liberty file for the given cells.

    Parameters
    ----------
    prj: BagProject
        BagProject object to be able to generate things
    lib_config : Mapping[str, Any]
        library configuration dictionary.
    sim_config : Mapping[str, Any]
        simulation configuration dictionary.
    cell_specs : Mapping[str, Any]
        cell specification dictionary.
    fake : bool
        True to generate fake liberty file.
    extract : bool
        True to run extraction.
    force_sim : bool
        True to force simulation runs.
    force_extract : bool
        True to force extraction runs.
    gen_all_env : bool
        True to generate liberty files for all environments.
    gen_sch : bool
        True to generate schematics.
    export_lay : bool
        Use CAD tool to export layout.
    log_level : LogLevel
        stdout logging level.
    """
    gen_specs_file: str = cell_specs['gen_specs_file']
    scenario: str = cell_specs.get('scenario', '')

    gen_specs: Mapping[str, Any] = read_yaml(gen_specs_file)
    impl_lib: str = gen_specs['impl_lib']
    impl_cell: str = gen_specs['impl_cell']
    lay_cls: str = gen_specs['lay_class']
    dut_params: Mapping[str, Any] = gen_specs['params']
    name_prefix: str = gen_specs.get('name_prefix', '')
    name_suffix: str = gen_specs.get('name_suffix', '')
    gen_root_dir: Path = Path(gen_specs['root_dir'])
    lib_root_dir = gen_root_dir / 'lib_gen'

    sim_precision: int = sim_config['precision']

    dsn_options = dict(
        extract=extract,
        force_extract=force_extract,
        gen_sch=gen_sch,
        log_level=log_level,
    )
    log_file = str(lib_root_dir / 'lib_gen.log')
    sim_db = prj.make_sim_db(lib_root_dir / 'dsn', log_file, impl_lib, dsn_options=dsn_options,
                             force_sim=force_sim, precision=sim_precision, log_level=log_level)
    dut = await sim_db.async_new_design(impl_cell, lay_cls, dut_params, export_lay=export_lay,
                                        name_prefix=name_prefix, name_suffix=name_suffix)

    environments: Mapping[str, Any] = lib_config['environments']
    nom_voltage_type: str = environments['nom_voltage_type']
    name_format: str = environments['name_format']
    voltage_precision: int = environments['voltage_precision']
    sim_env_list: List[Mapping[str, Any]] = environments['sim_envs']
    if not gen_all_env:
        sim_env_list = [sim_env_list[0]]

    voltage_fmt = '{:.%df}' % voltage_precision
    lib_file_base_name = f'{impl_cell}_{scenario}' if scenario else impl_cell
    for sim_env_config in sim_env_list:
        sim_env: str = sim_env_config['sim_env']
        voltages: Mapping[str, float] = sim_env_config['voltages']

        vstr_table = {k: voltage_fmt.format(v).replace('.', 'p') for k, v in voltages.items()}
        sim_env_name = name_format.format(sim_env=sim_env, **vstr_table)
        lib_file_name = f'{lib_file_base_name}_{sim_env_name}'

        cur_lib_config = dict(**lib_config)
        cur_lib_config.pop('environments')

        # setup lib config
        temperature = get_corner_temp(sim_env)[1]
        env_config = dict(
            name=sim_env_name,
            bag_name=sim_env,
            process=1.0,
            temperature=temperature,
            voltage=voltages[nom_voltage_type],
        )

        cur_lib_config['voltages'] = voltages
        cur_lib_config['sim_envs'] = [env_config]
        lib = Library(f'{impl_cell}_{sim_env_name}', cur_lib_config)

        out_file = gen_root_dir / f'{lib_file_name}.lib'
        lib_data, mm_specs, cur_work_dir = get_cell_info(lib, impl_cell, cell_specs,
                                                         lib_root_dir, voltage_fmt)

        mm_specs['fake'] = fake
        mm_specs['sim_env_name'] = sim_env_name
        for key in ['tran_tbm_specs', 'buf_params', 'in_cap_search_params', 'out_cap_search_params',
                    'seq_search_params', 'seq_delay_thres']:
            mm_specs[key] = sim_config[key]
        mm = sim_db.make_mm(LibertyCharMM, mm_specs)

        sim_db.log(f'Characterizing {lib_file_name}.lib')
        char_results = await sim_db.async_simulate_mm_obj('lib_char', cur_work_dir, dut, mm)
        pin_data = char_results.data

        _add_cell(lib, lib_data, pin_data)
        lib.generate(out_file)


def get_cell_info(lib: Library, impl_cell: str, cell_specs: Mapping[str, Any], lib_root_dir: Path,
                  voltage_fmt: str) -> Tuple[Mapping[str, Any], Dict[str, Any], Path]:
    sim_envs = lib.sim_envs
    if len(sim_envs) != 1:
        raise ValueError('Only support one corner per liberty file.')

    in_cap_range_scale: float = cell_specs['input_cap_range_scale']
    in_cap_min: float = cell_specs.get('input_cap_min', 1.0e-15)
    in_cap_guess: float = cell_specs.get('input_cap_guess', in_cap_min)
    out_min_fanout: float = cell_specs.get('min_fanout', 0.5)

    stdcell_pwr_pins: Sequence[str] = cell_specs.get('stdcell_pwr_pins', [])
    input_pins: Sequence[Mapping[str, Any]] = cell_specs.get('input_pins', [])
    output_pins: Sequence[Mapping[str, Any]] = cell_specs.get('output_pins', [])
    inout_pins: Sequence[Mapping[str, Any]] = cell_specs.get('inout_pins', [])
    pwr_pins: Mapping[str, str] = cell_specs['pwr_pins']
    gnd_pins: Mapping[str, str] = cell_specs['gnd_pins']

    in_defaults: Mapping[str, Any] = cell_specs.get('input_pins_defaults', {})
    out_defaults: Mapping[str, Any] = cell_specs.get('output_pins_defaults', {})
    inout_defaults: Mapping[str, Any] = cell_specs.get('inout_pins_defaults', {})

    pin_values: Mapping[str, int] = cell_specs.get('cond_defaults', {})

    seq_timing: Mapping[str, Any] = cell_specs.get('seq_timing', {})

    cell_props: Mapping[str, Any] = cell_specs.get('props', {})
    diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]] = cell_props.get('pin_opposite', [])

    custom_meas: Mapping[str, Mapping[str, Any]] = cell_specs.get('custom_measurements', {})

    # get supply values
    sup_values: Dict[str, float] = {}
    for pin, vtype in chain(pwr_pins.items(), gnd_pins.items()):
        sup_values[pin] = lib.get_voltage(vtype)

    # gather pin information
    pwr_domain: Dict[str, Tuple[str, str]] = {}
    reset_table: Dict[str, bool] = {}
    in_cap_table: Dict[str, float] = {}
    in_pin_list = _get_pin_info_list(input_pins, in_defaults, pwr_domain, reset_table,
                                     in_cap_table, in_cap_guess)
    io_pin_list = _get_pin_info_list(inout_pins, inout_defaults, pwr_domain)
    out_pin_list = _get_pin_info_list(output_pins, out_defaults, pwr_domain)

    lut_delay = lib.get_lut(LUTType.DELAY)
    delay_shape = lut_delay.shape
    delay_swp_info = lut_delay.get_swp_info(dict(trf_src='t_rf', cload='c_load'))
    seq_swp_info = lut_delay.get_swp_info(dict(trf_src='t_clk_rf', cload='c_load'))

    lut_cons = lib.get_lut(LUTType.CONSTRAINT)
    t_rf_list = lut_cons['trf_in']
    t_clk_rf_list = lut_cons['trf_src']
    cons_swp_order = lut_cons.get_swp_order(dict(trf_src='t_clk_rf', trf_in='t_rf'))

    out_io_info_table = dict(chain(_pin_info_iter(out_pin_list), _pin_info_iter(io_pin_list)))
    out_cap_num_freq = len(lib.get_lut(LUTType.MAX_CAP)['freq'])
    mm_specs = dict(
        sim_envs=sim_envs,
        thres_lo=lib.thres_lo,
        thres_hi=lib.thres_hi,
        in_cap_min_default=in_cap_min,
        in_cap_range_scale=in_cap_range_scale,
        out_max_trf=lib.get_max_input_transition(LogicType.COMB, is_clock=False),
        out_min_fanout=out_min_fanout,
        out_cap_num_freq=out_cap_num_freq,
        in_cap_table=in_cap_table,
        out_io_info_table=out_io_info_table,
        seq_timing=seq_timing,
        custom_meas=custom_meas,

        delay_shape=delay_shape,
        delay_swp_info=delay_swp_info,
        seq_shape=delay_shape,
        seq_swp_info=seq_swp_info,
        t_rf_list=t_rf_list,
        t_clk_rf_list=t_clk_rf_list,
        t_clk_rf_first=(cons_swp_order[0] == 't_clk_rf'),

        dut_info=dict(
            pwr_domain=pwr_domain,
            sup_values=sup_values,
            pin_values=pin_values,
            reset_list=list(reset_table.items()),
            diff_list=diff_list,
        ),
    )

    # get working directory
    vstr_table = {k: voltage_fmt.format(v).replace('.', 'p') for k, v in sup_values.items()}
    voltage_string = '_'.join((f'{k}_{vstr_table[k]}' for k in sorted(vstr_table.keys())))
    work_dir = lib_root_dir / sim_envs[0] / voltage_string

    # construct result dictionary
    lib_data = {k: cell_specs[k] for k in ['props', 'pwr_pins', 'gnd_pins']}
    lib_data['name'] = impl_cell
    lib_data['stdcell_pwr_pins'] = stdcell_pwr_pins
    lib_data['input_pins'] = in_pin_list
    lib_data['output_pins'] = out_pin_list
    lib_data['inout_pins'] = io_pin_list
    return lib_data, mm_specs, work_dir


def _pin_info_iter(pin_list: List[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for pin_info in pin_list:
        basename: str = pin_info.get('basename', '')

        if basename:
            # bus pin
            bus_range = pin_info['bus_range']
            values: List[Dict[str, Any]] = pin_info['values']

            for bus_idx, cur_info in zip(bus_range, values):
                pin_name = get_bus_bit_name(basename, bus_idx, cdba=True)
                yield pin_name, cur_info
        else:
            # scalar pin
            yield pin_info['name'], pin_info


def _get_pin_info_list(src_list: Sequence[Mapping[str, Any]], defaults: Mapping[str, Any],
                       pwr_domain: Dict[str, Tuple[str, str]],
                       reset_table: Optional[Dict[str, bool]] = None,
                       in_cap_table: Optional[Dict[str, float]] = None, cap_guess: float = 1.0e-15
                       ) -> List[Dict[str, Any]]:
    pin_list = []
    empty_dict = {}
    default_dict = {k: v for k, v in defaults.items()}
    for pin_info in src_list:
        pin_name: str = pin_info['name']
        reset_val: Optional[int] = pin_info.get('reset_val', None)

        basename, bus_range = parse_cdba_name(pin_name)
        if bus_range is None:
            # scalar pin
            if 'basename' in pin_info:
                raise ValueError('Scalar pins cannot have basename entry defined.')

            cur_info = default_dict.copy()
            cur_info.update(pin_info)
            cur_info.pop('hide', None)
            in_cap_guess = cur_info.pop('cap_guess', cap_guess)

            if in_cap_table is not None:
                in_cap_table[pin_name] = in_cap_guess

            # record power domain
            pwr_domain[pin_name] = (cur_info['gnd_pin'], cur_info['pwr_pin'])
            if reset_val is not None:
                if reset_table is None:
                    raise ValueError('reset_table is not given but reset_val is defined.')
                if pin_name in reset_table:
                    raise ValueError(f'pin {pin_name} is already in reset_table.')
                reset_table[pin_name] = (reset_val == 1)

            if not pin_info.get('hide', False):
                pin_list.append(cur_info)
        else:
            # bus pin
            values: Optional[List[Dict[str, Any]]] = pin_info.get('values', None)
            bus_defaults: Dict[str, Any] = pin_info.get('defaults', empty_dict)

            cur_defaults = default_dict.copy()
            cur_defaults.update(bus_defaults)
            if 'reset_val' not in cur_defaults:
                cur_defaults['reset_val'] = None
            num_bits = len(bus_range)
            # NOTE: make dictionary copies, so we can add cap info to them later
            if values is None:
                values = [cur_defaults.copy() for _ in range(num_bits)]
            elif len(values) != num_bits:
                raise ValueError(f'values list of bus {pin_name} length mismatch')
            else:
                values = []
                for val_ in values:
                    val_ = val_.copy()
                    val_.update(cur_defaults)
                    values.append(val_)

            # record power domain and reset values
            for bus_idx, bit_info in zip(bus_range, values):
                pwr_str: str = _get('pwr_pin', bit_info, cur_defaults)
                gnd_str: str = _get('gnd_pin', bit_info, cur_defaults)
                reset_val: Optional[int] = _get('reset_val', bit_info, cur_defaults)

                in_cap_guess = bit_info.pop('cap_guess', cap_guess)

                bit_name = get_bus_bit_name(basename, bus_idx, cdba=True)
                pwr_domain[bit_name] = (gnd_str, pwr_str)
                if reset_val is not None:
                    if reset_table is None:
                        raise ValueError('reset_table is not given but reset_val is defined.')
                    if bit_name in reset_table:
                        raise ValueError(f'pin {bit_name} is already in reset_table.')
                    reset_table[bit_name] = (reset_val == 1)

                if in_cap_table is not None:
                    in_cap_table[bit_name] = in_cap_guess

            if not pin_info.get('hide', False):
                pin_list.append(dict(name=pin_name, basename=basename, bus_range=bus_range,
                                     values=values))
    return pin_list


def _get(key: str, dict1: Mapping[str, Any], dict2: Mapping[str, Any]) -> Any:
    ans = dict1.get(key, None)
    if ans is not None:
        return ans
    return dict2[key]


def _add_cell(lib: Library, cell_info: Mapping[str, Any], pin_data: Mapping[str, Mapping[str, Any]]
              ) -> None:
    empty_list = []

    name: str = cell_info['name']
    props: Dict[str, Any] = cell_info['props']
    pwr_pins: Dict[str, str] = cell_info['pwr_pins']
    gnd_pins: Dict[str, str] = cell_info['gnd_pins']
    stdcell_pwr_pins: List[str] = cell_info.get('stdcell_pwr_pins', [])
    input_pins: List[Dict[str, Any]] = cell_info.get('input_pins', empty_list)
    output_pins: List[Dict[str, Any]] = cell_info.get('output_pins', empty_list)
    inout_pins: List[Dict[str, Any]] = cell_info.get('inout_pins', empty_list)

    cell = lib.create_cell(name, pwr_pins, gnd_pins, props, stdcell_pwr_pins=stdcell_pwr_pins)
    _add_pins(cell, TermType.input, input_pins, pin_data)
    _add_pins(cell, TermType.output, output_pins, pin_data)
    _add_pins(cell, TermType.inout, inout_pins, pin_data)


def _add_pins(cell: Cell, pin_type: TermType, pin_list: List[Dict[str, Any]],
              pin_data: Mapping[str, Mapping[str, Any]]) -> None:
    remove_keys = ['reset_val', 'cap_info', 'timing_info']
    for pin_info in pin_list:
        basename: str = pin_info.get('basename', '')

        if basename:
            # bus pin
            bus_range = pin_info['bus_range']
            values: List[Dict[str, Any]] = pin_info['values']
            bus = cell.create_bus(pin_info['name'], pin_type)

            for idx, (bus_idx, cur_info) in enumerate(zip(bus_range, values)):
                bit_name = get_bus_bit_name(basename, bus_idx, cdba=True)
                for rm_key in remove_keys:
                    cur_info.pop(rm_key, None)

                cur_info.update(pin_data[bit_name])
                timing_list: Optional[List[Dict[str, Any]]] = cur_info.pop('timing', None)
                pin = bus.create_pin(idx, pin_type, cur_info)
                if timing_list is not None:
                    for timing in timing_list:
                        pin.add_timing(**timing)
        else:
            # scalar pin
            pin_name = pin_info['name']
            for rm_key in remove_keys:
                pin_info.pop(rm_key, None)

            pin_info.update(pin_data[pin_name])
            timing_list: Optional[List[Dict[str, Any]]] = pin_info.pop('timing', None)

            pin = cell.create_pin(pin_type, pin_info)
            if timing_list is not None:
                for timing in timing_list:
                    pin.add_timing(**timing)

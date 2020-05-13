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

from typing import Dict, Any, Sequence, Optional, Callable, List, Mapping

import json
import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_testbenches__digital_tb_tran(Module):
    """Schematic generator for transient simulation of digital blocks.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'digital_tb_tran.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            dut_lib='Transistor DUT library name.',
            dut_cell='Transistor DUT cell name.',
            in_file_list='input PWL waveform file list.',
            clk_file_list='clk PWL waveform file list.',
            load_list='output load capacitance list.',
            vbias_list='List of voltage biases.',
            src_list='List of other sources.',
            dut_conns='DUT connection dictionary.',
            dut_params='DUT design parameters.',
            no_conns='List of non-connected nets',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            load_list=None,
            vbias_list=None,
            dut_conns={},
            dut_params=None,
            no_conns=None,
            in_file_list=[],
            clk_file_list=[],
            src_list=[],
        )

    def design(self, dut_lib: str, dut_cell: str,
               in_file_list: Sequence[Sequence[str]],
               clk_file_list: Sequence[Sequence[str]],
               load_list: Optional[Sequence[Sequence[str]]],
               vbias_list: Optional[Sequence[Sequence[str]]],
               dut_conns: Dict[str, str],
               dut_params: Optional[Param],
               no_conns: Sequence[str],
               src_list: Sequence[Mapping[str, Any]]) -> None:
        """Design the testbench.

        The elements of parameter lists are either (pos_term, param) or
        (pos_term, neg_term, param), where pos_term/neg_term are the positive/negative
        terminals of the voltage sources or capacitors.  The negative terminal
        defaults to VSS if not specified.

        for ``load_list`` and ``vbias_list``, if None is given (the default), then
        the default load/bias voltages will be used (the ones shown in schematic
        template).  If an empty list is given, then they'll be removed entirely.

        Parameters
        ----------
        dut_lib : str
            DUT library name
        dut_cell : str
            DUT cell name
        in_file_list : Sequence[Sequence[str]]
            List of PWL input stimuli files
        clk_file_list : Sequence[Sequence[str]]
            List of PWL clk stimuli files
        load_list : Optional[Sequence[Sequence[str]]]
            List of ideal capacitor loads
        vbias_list : Optional[Sequence[Sequence[str]]]
            List of voltage biases
        dut_conns : Dict[str, str]
            DUT connection dictionary
        dut_params: Optional[Param]
            Replace the DUT statically if empty, otherwise call design with dut_params.
        no_conns: List[str]
            Connects the content of this list to noConn.
        src_list : Sequence[Mapping[str, Any]]
            list of sources and loads.
        """

        if no_conns:
            self.rename_instance('XNC', f'XNC<{len(no_conns) - 1}:0>',
                                 [('noConn', ','.join(no_conns))])
        else:
            self.delete_instance('XNC')

        if vbias_list is None:
            vbias_list = [('VDD', 'vdd')]

        # combine src_list and load_list
        src_load_list = list(src_list)
        if load_list:
            for cap_info in load_list:
                if len(cap_info) == 2:
                    pos_term, val = cap_info
                    neg_term = 'VSS'
                elif len(cap_info) == 3:
                    pos_term, neg_term, val = cap_info
                else:
                    raise ValueError(f'Cannot parse cap element: {cap_info}')

                src_load_list.append(dict(type='cap', lib='analogLib', value=val,
                                          conns=dict(PLUS=pos_term, MINUS=neg_term)))

        # setup DUT
        dut_static = dut_params is None
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=dut_static,
                                     keep_connections=True)
        if not dut_static:
            self.instances['XDUT'].design(**dut_params)
        self.reconnect_instance('XDUT', ((k, v) for k, v in dut_conns.items()))

        # setup PWL files
        def get_path_str(fname: str) -> str:
            return json.dumps(str(Path(fname).resolve()))

        self._array_and_set_params('VIN', in_file_list, 'fileName', get_path_str)
        self._array_and_set_params('VCLK', clk_file_list, 'fileName', get_path_str)
        # setup voltage biases
        self._array_and_set_params('VSUP', vbias_list, 'vdc', None)

        # setup sources and loads
        self.design_sources_and_loads(src_load_list, default_name='CLOAD')

    def _array_and_set_params(self, inst_name: str, info_list: Sequence[Sequence[str]],
                              param_name: str, fun: Optional[Callable[[str], str]]) -> None:
        if info_list:
            inst_term_list = []
            param_list = []
            for ele in info_list:
                if len(ele) == 2:
                    pos_term = ele[0]
                    neg_term = 'VSS'
                    val = ele[1]
                elif len(ele) == 3:
                    pos_term = ele[0]
                    neg_term = ele[1]
                    val = ele[2]
                else:
                    raise ValueError(f'Cannot parse list element: {ele}')
                cur_name = f'X{pos_term.upper()}'
                inst_term_list.append((cur_name, [('PLUS', pos_term), ('MINUS', neg_term)]))
                param_list.append(val if fun is None else fun(val))

            self.array_instance(inst_name, inst_term_list=inst_term_list)
            for (name, _), param in zip(inst_term_list, param_list):
                self.instances[name].set_param(param_name, param)
        else:
            self.remove_instance(inst_name)

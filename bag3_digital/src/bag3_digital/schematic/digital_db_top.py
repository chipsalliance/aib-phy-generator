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

# -*- coding: utf-8 -*-

from typing import Dict, Any, Sequence, Tuple, Optional, Mapping

import pkg_resources
from pathlib import Path

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__digital_db_top(Module):
    """Module for library bag3_digital cell digital_db_top.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'digital_db_top.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            buf_params='Buffer parameters.',
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            dut_conns='DUT connection dictionary.',
            source_load_params='Source load params for keeping wrapper interface common',
            dut_params='DUT design parameters.',
            dut_load='Flag for using the DUT as a load',
            dut_m='How many of the dut to use as a load',
            dut_load_conns='DUT load connections',
            in_pin_list='Input pin list for exporting internal signals as inputs',
            out_pin_list='Output pin list for exporting internal signals as outputs',
            sup_pin_list='Supply pin list.',
            no_conns='List of non-connected nets',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dut_conns={},
            dut_params=None,
            source_load_params=None,
            dut_load=False,
            dut_m=0,
            dut_load_conns={},
            in_pin_list=None,
            out_pin_list=None,
            sup_pin_list=None,
            no_conns=[],
        )

    def design(self, buf_params: Sequence[Tuple[Any, ...]], dut_lib: str, dut_cell: str,
               in_pin_list: Optional[Sequence[str]], out_pin_list: Optional[Sequence[str]],
               sup_pin_list: Optional[Sequence[str]], dut_conns: Mapping[str, str],
               dut_params: Optional[Param], dut_load: bool, dut_m: int,
               source_load_params: Optional[Sequence[Mapping[str, Any]]],
               dut_load_conns: Mapping[str, Any], no_conns: Sequence[str]) -> None:

        self.design_sources_and_loads(source_load_params)
        if buf_params:
            array_inst_names = ['XBUF%s' % x for x in range(0, len(buf_params))]
            self.array_instance('XBUF', inst_name_list=array_inst_names)
            for entry, name in zip(buf_params, array_inst_names):
                input_net = ''
                output_net = ''
                has_input = False
                has_output = False
                cur_length = len(entry)
                if cur_length == 1:
                    cur_params = entry[0]
                elif cur_length == 2:
                    cur_params, input_net = entry
                    has_input = True
                elif cur_length == 3:
                    cur_params, input_net, output_net = entry
                    has_input = True
                    has_output = True
                else:
                    raise ValueError('In design method of digital_db_top, '
                                     f'entry is too long! {entry}')
                self.instances[name].design(**cur_params)
                if has_input:
                    self.reconnect_instance_terminal(name, 'in', input_net)
                    if has_output:
                        if isinstance(output_net, Mapping):
                            self.reconnect_instance(name, ((k, v) for k, v in output_net.items()))
                        else:
                            self.reconnect_instance_terminal(name, 'out', output_net)
        else:
            self.delete_instance('XBUF')

        dut_static = dut_params is None
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=dut_static,
                                     keep_connections=True)
        if not dut_static:
            self.instances['XDUT'].design(**dut_params)
        if dut_load:
            # need to use the DUT as a load for itself
            dut_name_list = ['XDUT%s' % x for x in range(0, dut_m + 1)]
            self.array_instance('XDUT', dut_name_list)
            self.reconnect_instance('XDUT0', ((k, v) for k, v in dut_conns.items()))
            for i in range(1, dut_m + 1):
                self.reconnect_instance('XDUT%s' % i, ((k, v) for k, v in dut_load_conns.items()))
        else:
            self.reconnect_instance('XDUT', ((k, v) for k, v in dut_conns.items()))

        if in_pin_list is not None:
            has_in = False
            for cur_in in in_pin_list:
                if cur_in == 'in':
                    has_in = True
                else:
                    self.add_pin(cur_in, TermType.input)
            if not has_in:
                self.remove_pin('in')

        if out_pin_list is not None:
            has_out = False
            for cur_out in out_pin_list:
                if cur_out == 'out':
                    has_out = True
                else:
                    self.add_pin(cur_out, TermType.output)
            if not has_out:
                self.remove_pin('out')

        if sup_pin_list is not None:
            has_vdd = has_vss = False
            for cur_sup in sup_pin_list:
                if cur_sup == 'VDD':
                    has_vdd = True
                elif cur_sup == 'VSS':
                    has_vss = True
                else:
                    self.add_pin(cur_sup, TermType.inout)
            if not has_vdd:
                self.remove_pin('VDD')
            if not has_vss:
                self.remove_pin('VSS')

        if no_conns:
            self.rename_instance('Xnc', f'Xnc<{len(no_conns) - 1}:0>',
                                 [('noConn', ','.join(no_conns))])
        else:
            self.delete_instance('Xnc')

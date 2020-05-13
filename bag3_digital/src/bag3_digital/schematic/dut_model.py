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

from typing import Dict, Any, List, Union, Tuple, Type

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param
from pybag.enum import TermType

PinListType = List[Union[str, Tuple[str, int]]]


# noinspection PyPep8Naming
class bag3_digital__dut_model(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'dut_model.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
        self.pin_lut = {
            TermType.input: 'in',
            TermType.output: 'out',
            TermType.inout: 'inout'
        }
        self.pin_list = []
        self.pin_count = 0

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(inout='inout pin list',
                    input='input pin list',
                    output='output pin list')

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(inout=None, input=None, output=None)

    def _configure_pins(self, pin_list: PinListType, ttype: TermType) -> None:
        keep_default = False
        default_pin = self.pin_lut[ttype]
        if pin_list is None:
            self.remove_pin(default_pin)
            return
        for pin in pin_list:
            if isinstance(pin, tuple):
                pin_str, num = pin
                pin_name = f'{pin_str}<{num - 1}:0>' if num > 1 else pin_str
            elif isinstance(pin, str):
                pin_name = pin
                num = 1
            else:
                raise ValueError(f'unsupported type {type(pin)}')
            if pin_name != default_pin:
                self.add_pin(pin_name, ttype)
                self.pin_list.append(pin_name)
            else:
                keep_default = True
            self.pin_count += num
        if not keep_default:
            self.remove_pin(default_pin)

    def design(self, input: PinListType, output: PinListType, inout: PinListType) -> None:
        self._configure_pins(input, TermType.input)
        self._configure_pins(output, TermType.output)
        self._configure_pins(inout, TermType.inout)

        num = self.pin_count
        nc_name = f'Xnc<{num - 1}:0>' if num > 1 else 'Xnc'
        self.rename_instance('Xnc', nc_name, [('noConn', ','.join(self.pin_list))])

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

from typing import Dict, Any, List

import pkg_resources
from pathlib import Path

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__se_to_diff(Module):
    """Module for library bag3_digital cell se_to_diff.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'se_to_diff.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            invp_params_list='Positive output chain parameters.',
            invn_params_list='Negative output chain parameters.',
            pg_params='passgate parameters.',
            export_pins='True to export simulation pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(export_pins=False)

    def design(self, invp_params_list: List[Param], invn_params_list: List[Param],
               pg_params: Param, export_pins: bool) -> None:
        if len(invp_params_list) != 2 or len(invn_params_list) != 3:
            raise ValueError('Wrong number of parameters for inverters.')

        self.instances['XINVP0'].design(**invp_params_list[0])
        self.instances['XINVP1'].design(**invp_params_list[1])
        self.instances['XINVN0'].design(**invn_params_list[0])
        self.instances['XINVN1'].design(**invn_params_list[1])
        self.instances['XINVN2'].design(**invn_params_list[2])
        self.instances['XPASS'].design(**pg_params)

        if export_pins:
            self.add_pin('midn_pass0', TermType.output)
            self.add_pin('midn_pass1', TermType.output)
            self.add_pin('midn_inv', TermType.output)
            self.add_pin('midp', TermType.output)

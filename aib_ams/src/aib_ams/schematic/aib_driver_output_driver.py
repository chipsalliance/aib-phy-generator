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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class aib_ams__aib_driver_output_driver(Module):
    """Module for library aib_ams cell aib_driver_output_driver.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_driver_output_driver.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            unit_params='output unit cell parameters',
            pupd_params='weak pull up pull down parameters',
        )

    def design(self, unit_params: Param, pupd_params: Param) -> None:
        self.instances['XUNIT<5:0>'].design(**unit_params)
        self.instances['XPUPD'].design(**pupd_params.copy(append=dict(strong=False)))
        self.instances['XSUM'].design(nin=7)
        self.reconnect_instance_terminal('XSUM', 'in<6:0>', 'txpadout_tmp<6:0>')

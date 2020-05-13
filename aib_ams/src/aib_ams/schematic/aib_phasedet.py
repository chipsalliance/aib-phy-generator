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
class aib_ams__aib_phasedet(Module):
    """Module for library aib_ams cell aib_phasedet.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_phasedet.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            se_params='two-three splitter parameters.',
            flop_params='strongarm flop parameters.',
            inv_params='dummy inverter parameters.',
        )

    def design(self, se_params: Param, flop_params: Param, inv_params: Param) -> None:
        self.instances['XSED'].design(**se_params)
        self.instances['XSEU'].design(**se_params)
        self.instances['XFLOPD'].design(**flop_params)
        self.instances['XFLOPU'].design(**flop_params)
        self.instances['XDUMD'].design(**inv_params)
        self.instances['XDUMU'].design(**inv_params)

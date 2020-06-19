# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
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
class aib_ams__aib_txanlg_core(Module):
    """Module for library aib_ams cell aib_txanlg_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_txanlg_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            drv_params='output driver parameters.',
            data_lv_params='data level shifter parameters.',
            ctrl_lv_params='control signals level shifter parameters.',
        )

    def design(self, drv_params: Param, data_lv_params: Param, ctrl_lv_params: Param) -> None:
        self.instances['XDRV'].design(**drv_params)
        self.instances['XLV_DIN'].design(dual_output=False, **data_lv_params)
        self.instances['XLV_ITX_EN'].design(dual_output=True, **ctrl_lv_params)
        self.instances['XLV_PD'].design(dual_output=False, **ctrl_lv_params)
        self.instances['XLV_PU'].design(dual_output=False, **ctrl_lv_params)
        self.instances['XLV_PDRV<1:0>'].design(dual_output=False, **ctrl_lv_params)
        self.instances['XLV_NDRV<1:0>'].design(dual_output=False, invert_out=True,
                                               **ctrl_lv_params)

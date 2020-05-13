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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class aib_ams__aib_rxanlg_core(Module):
    """Module for library aib_ams cell aib_rxanlg_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_rxanlg_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            data_params='data single-to-diff parameters.',
            clk_params='clock single-to-diff-match parameters.',
            data_lv_params='data level shifter parameters.',
            ctrl_lv_params='control level shifter parameters.',
            por_lv_params='power-on-reset level shifter parameters.',
            buf_ctrl_lv_params='ctrl level shifter input buffer parameters.',
            buf_por_lv_params='por level shifter input buffer parameters.',
            inv_params='asnyc output inverter parameters.',
        )

    def design(self, data_params: Param, clk_params: Param, data_lv_params: Param,
               ctrl_lv_params: Param, por_lv_params: Param, buf_ctrl_lv_params: Param,
               buf_por_lv_params: Param, inv_params: Param) -> None:

        self.instances['XSE_DATA'].design(**data_params)
        self.instances['XSE_CLK'].design(**clk_params)
        self.instances['XLV_DATA'].design(dual_output=True, **data_lv_params)
        self.instances['XLV_CLK'].design(dual_output=True, **data_lv_params)
        self.instances['XINV'].design(**inv_params)
        self.instances['XDUM'].design(**inv_params)
        self.instances['XLV_CLK_EN'].design(dual_output=True, lev_params=ctrl_lv_params,
                                            buf_params=buf_ctrl_lv_params)
        self.instances['XLV_DATA_EN'].design(dual_output=True, lev_params=ctrl_lv_params,
                                             buf_params=buf_ctrl_lv_params)
        self.instances['XPOR'].design(dual_output=True, **buf_por_lv_params)
        self.instances['XPOR_DUM'].design(dual_output=True, **buf_por_lv_params)
        self.instances['XLV_POR'].design(dual_output=True, **por_lv_params)
        self.instances['XLV_DUM'].design(dual_output=True, **por_lv_params)

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
class bag3_digital__flop_strongarm(Module):
    """Module for library bag3_digital cell flop_strongarm.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'flop_strongarm.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sa_params='strongarm frontend parameters.',
            sr_params='sr latch parameters.',
            has_rstlb='True to add rstlb functionality.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(has_rstlb=False)

    def design(self, sa_params: Param, sr_params: Param, has_rstlb: bool) -> None:
        inbuf_test = sr_params.get('inbuf_params', None)
        if inbuf_test is None:
            raise ValueError('SR latch must have input buffers.')

        self.instances['XSA'].design(has_rstb=has_rstlb, **sa_params)
        self.instances['XSR'].design(has_rstb=has_rstlb, **sr_params)

        if not has_rstlb:
            self.remove_pin('rstlb')

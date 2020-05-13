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

from typing import Dict, Any, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__sr_latch_symmetric(Module):
    """Module for library bag3_digital cell sr_latch_symmetric.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sr_latch_symmetric.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            core_params='SR latch core parameters.',
            outbuf_params='output buffer parameters.',
            inbuf_params='s/r input buffer parameters.',
            has_rstb='True to enable rstb functionality.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(outbuf_params=None, inbuf_params=None, has_rstb=False)

    def design(self, core_params: Param, outbuf_params: Optional[Param],
               inbuf_params: Optional[Param], has_rstb: bool) -> None:
        inst = self.instances['XCORE']
        inst.design(has_rstb=has_rstb, **core_params)

        if not has_rstb:
            self.remove_pin('rstlb')
            self.remove_pin('rsthb')

        if outbuf_params is None:
            self.remove_instance('XOBUF<1:0>')
            self.reconnect_instance('XCORE', [('q', 'q'), ('qb', 'qb')])
        else:
            self.instances['XOBUF<1:0>'].design(**outbuf_params)

        if inbuf_params is None:
            self.remove_instance('XIBUF<1:0>')
        else:
            self.remove_pin('s')
            self.remove_pin('r')
            self.instances['XIBUF<1:0>'].design(**inbuf_params)
            self.reconnect_instance_terminal('XIBUF<1:0>', 'out', 's,r')

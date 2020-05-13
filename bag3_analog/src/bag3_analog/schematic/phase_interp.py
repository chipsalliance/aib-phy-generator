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

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_analog__phase_interp(Module):
    """Module for library bag3_analog cell phase_interp.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'phase_interp.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def is_leaf_model(cls) -> bool:
        return True

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tri_params='Tristate Inverter Params',
            inv_params='Output Inverter Params',
            nbits='number of control bits',
            export_outb='True to export input of output buffer',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            export_outb=False,
        )

    def design(self, tri_params: Param, inv_params: Param, nbits: int, export_outb: bool) -> None:
        if nbits < 2:
            raise ValueError('nbits must be >= 2')

        suffix = f'<{nbits - 1}:0>'
        for name in ['a', 'b']:
            basename = f'XINV{name.upper()}'
            inst_conns = [('en', name + '_en' + suffix), ('enb', name + '_enb' + suffix),
                          ('out', name + '_outb' + suffix)]
            new_name = basename + suffix
            self.rename_instance(basename, new_name, conn_list=inst_conns)
            self.instances[new_name].design(**tri_params)

        self.instances['XBUF'].design(**inv_params)
        self.instances['XSUM'].design(nin=2 * nbits)
        self.reconnect_instance_terminal('XSUM', f'in<{2 * nbits - 1}:0>',
                                         f'a_outb{suffix},b_outb{suffix}')

        for name in ['a_en', 'b_en', 'a_enb', 'b_enb']:
            self.rename_pin(name, name + suffix)

        if export_outb:
            self.add_pin('outb', TermType.output)

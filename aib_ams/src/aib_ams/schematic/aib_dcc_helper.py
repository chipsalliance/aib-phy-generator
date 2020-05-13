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
class aib_ams__aib_dcc_helper(Module):
    """Module for library aib_ams cell aib_dcc_helper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_dcc_helper.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            core_params='dcc_helper_core parameters.',
            sync_params='synchronizer flop parameters.',
            buf_params='clock buffer parameters.',
            nsync='number of synchronizer flops',
        )

    def design(self, core_params: Param, sync_params: Param, buf_params: Param, nsync: int) -> None:
        if nsync < 2:
            raise ValueError('nsync must be >= 2.')

        if nsync != 2:
            idx1 = nsync - 1
            suf = f'<{idx1}:0>'
            inst_name = 'XSYNC' + suf
            dum_name = 'XDUMSYNC' + suf
            inst_conns = [('outp', 'rstlb_p' + suf), ('outn', 'rstlb_n' + suf),
                          ('inp', f'rstlb_p<{nsync - 2}:0>,VDD'),
                          ('inn', f'rstlb_n<{nsync - 2}:0>,VSS')]
            dum_conns = [('outp', 'dump' + suf), ('outn', 'dumn' + suf),
                         ('inp', f'dump<{nsync - 2}:0>,VDD'),
                         ('inn', f'dumn<{nsync - 2}:0>,VSS')]
            self.rename_instance('XSYNC<1:0>', inst_name, conn_list=inst_conns)
            self.rename_instance('XDUMSYNC<1:0>', dum_name, conn_list=dum_conns)
            self.reconnect_instance_terminal('XCORE', 'rstlb', f'rstlb_p<{idx1}>')
            nc_net = f'unused,rstlb_n<{idx1}>,dump<{idx1}>,dumn<{idx1}>'
            self.reconnect_instance_terminal('XNC<3:0>', 'noConn', nc_net)
        else:
            inst_name = 'XSYNC<1:0>'
            dum_name = 'XDUMSYNC<1:0>'

        self.instances['XCORE'].design(**core_params)
        self.instances[inst_name].design(**sync_params)
        self.instances[dum_name].design(**sync_params)
        self.instances['XCKBUF'].design(dual_output=False, **buf_params)
        self.instances['XDUMBUF'].design(dual_output=False, **buf_params)

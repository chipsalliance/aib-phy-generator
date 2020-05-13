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
class aib_ams__aib_dlycell_no_flop(Module):
    """Module for library aib_ams cell aib_dlycell_no_flop.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_dlycell_no_flop.yaml')))

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
            bk_inv_params='Schematic parameters for bk inv',
            dc_core_params='Schematic parameters for delay cell core',
            num_core='Number of delay cell cores in one delay cell',
            is_dum='True if this is a dummy cell in the delay line',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            num_core=1,
            is_dum=False,
        )

    def design(self, bk_inv_params: Param, dc_core_params: Param, num_core: int, is_dum: bool
               ) -> None:
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        self.instances['XBKInv0'].design(**bk_inv_params)
        self.instances['XBKInv1'].design(**bk_inv_params)
        self.instances['XCore'].design(**dc_core_params)

        if num_core > 1:
            inst_suf = f'<{num_core - 1}:0>'
            if is_dum:
                if num_core > 2:
                    term_suf = f'<{num_core - 2}:0>'

                    # no connections
                    self.rename_instance('XNC_in', f'XNC_in{term_suf}',
                                         [('noConn', f'NC_in{term_suf}')])
                    self.rename_instance('XNC_out', f'XNC_out{term_suf}',
                                         [('noConn', f'NC_out{term_suf}')])
                    self.rename_instance('XNC_ci', f'XNC_ci{term_suf}',
                                         [('noConn', f'NC_ci{term_suf}')])
                    self.rename_instance('XNC_co', f'XNC_co{term_suf}',
                                         [('noConn', f'NC_co{term_suf}')])
                else:
                    term_suf = ''

                # terminal connections
                conn_list = [
                    ('in_p', f'in_p,NC_in{term_suf}'),
                    ('co_p', f'NC_co{term_suf},co_p'),
                    ('out_p', f'out_p,NC_out{term_suf}'),
                    ('ci_p', f'NC_ci{term_suf},ci_p'),
                ]
            else:
                term_suf = f'<{num_core - 2}:0>' if num_core > 2 else ''

                # terminal connections
                conn_list = [
                    ('in_p', f'in_p,mid_in{term_suf}'),
                    ('co_p', f'mid_in{term_suf},co_p'),
                    ('out_p', f'out_p,mid_out{term_suf}'),
                    ('ci_p', f'mid_out{term_suf},ci_p'),
                ]
            self.rename_instance('XCore', f'XCore{inst_suf}', conn_list)

        if not is_dum or num_core == 1:
            for inst in ['XNC_in', 'XNC_out', 'XNC_ci', 'XNC_co']:
                self.remove_instance(inst)

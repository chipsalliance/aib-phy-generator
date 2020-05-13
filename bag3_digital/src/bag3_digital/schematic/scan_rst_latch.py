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
class bag3_digital__scan_rst_latch(Module):
    """Module for library bag3_digital cell scan_rst_latch.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'scan_rst_latch.yaml')))

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
            tin='Schematic parameters of input tristate inverter',
            tfb='Schematic parameters of feedback tristate inverter',
            nor='Schematic parameters of nor',
            scan='True to enable scanability',
            pg='Schematic parameters of pass gate if scanable',
            inv='Schematic parameters of scan inverter if scanable',
            dual_output='True to export out and outb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            pg=None,
            inv=None,
            dual_output=True,
        )

    def design(self, tin: Param, tfb: Param, nor: Param, scan: bool, pg: Optional[Param],
               inv: Optional[Param], dual_output: bool) -> None:
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
        self.instances['XTBUF'].design(**tin)
        self.instances['XTFB'].design(**tfb)
        self.instances['XNOR'].design(**nor)
        if scan:
            self.instances['XInvScan'].design(**inv)
            self.instances['XTBUF_scan'].design(**tin)
            self.instances['XPG0'].design(**pg)
            self.instances['XPG1'].design(**pg)
            self.instances['XCM'].design(nin=3)
            self.reconnect_instance_terminal('XCM', 'in<2:0>', 'fb,inb1,scan_inb1')
        else:
            self.instances['XCM'].design(nin=2)
            self.reconnect_instance_terminal('XCM', 'in<1:0>', 'fb,inb')
            for inst in ['XInvScan', 'XTBUF_scan', 'XPG0', 'XPG1']:
                self.remove_instance(inst)
            for pin in ['scan_in', 'scan_en']:
                self.remove_pin(pin)

        if dual_output is False:
            self.remove_pin('outb')

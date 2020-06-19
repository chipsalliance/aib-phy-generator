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
from pybag.enum import TermType


# noinspection PyPep8Naming
class aib_ams__aib_dlycell_core(Module):
    """Module for library aib_ams cell aib_dlycell_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_dlycell_core.yaml')))

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
            in_params='Parameters for input NAND',
            sr0_params='Parameters for SR0 NAND',
            sr1_params='Parameters for SR1 NAND',
            out_params='Parameters for output NAND',
            feedback='True to connect ci_p and co_p',
            output_sr_pins='True to output sr1_o and sr0_o pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(feedback=False, output_sr_pins=False)

    def design(self, in_params: Param, sr0_params: Param, sr1_params: Param,
               out_params: Param, feedback: bool, output_sr_pins: bool) -> None:
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
        self.instances['XNAND_in'].design(**in_params)
        self.instances['XNAND_out'].design(**out_params)
        self.instances['XNAND_SR0'].design(**sr0_params)
        self.instances['XNAND_SR1'].design(**sr1_params)

        if feedback:
            for pin in ['ci_p', 'co_p']:
                self.remove_pin(pin)
            self.reconnect_instance_terminal('XNAND_in', 'out', 'fb')
            self.reconnect_instance_terminal('XNAND_out', 'in<1:0>', 'sr1_o,fb')

        if output_sr_pins:
            self.add_pin('sr1_o', TermType.output)
            self.add_pin('sr0_o', TermType.output)

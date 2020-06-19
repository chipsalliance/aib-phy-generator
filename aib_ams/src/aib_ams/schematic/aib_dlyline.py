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

# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class aib_ams__aib_dlyline(Module):
    """Module for library aib_ams cell aib_dlyline.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_dlyline.yaml')))

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
            dlycell_params='Delay Cell parameters',
            num_insts='Number of instances of delay cells',
            num_dum='Number of instances of dummy delay cells',
            flop='True to have flops in delay cell',
            flop_char='True to add flop characterization pins.',
            output_sr_pins='True to output measurement pins.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            num_dum=0,
            flop=True,
            flop_char=False,
            output_sr_pins=False,
        )

    def design(self, dlycell_params: Param, num_insts: int, num_dum: int, flop: bool,
               flop_char: bool, output_sr_pins: bool) -> None:
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
        if not flop:
            self.replace_instance_master('XCELL', 'aib_ams', 'aib_dlycell_no_flop',
                                         keep_connections=True)
            self.replace_instance_master('XDUM', 'aib_ams', 'aib_dlycell_no_flop',
                                         keep_connections=True)
            for name in ['RSTb', 'CLKIN', 'iSI', 'SOOUT', 'iSE']:
                self.remove_pin(name)

        self.instances['XCELL'].design(**dlycell_params)

        if num_insts > 2:

            if output_sr_pins:
                conn_list = [
                    ('in_p', f'a<{num_insts - 2}:0>,dlyin'),
                    ('bk', f'bk<{num_insts - 1}:0>'),
                    ('ci_p', f'b<{num_insts - 1}:0>'),
                    ('out_p', f'b<{num_insts-2}:0>,dlyout'),
                    ('co_p', f'a<{num_insts - 1}:0>'),
                    ('si', f'so<{num_insts-2}:0>,iSI'),
                    ('so', f'SOOUT,so<{num_insts-2}:0>'),
                    ('srqb', f'srqb<{num_insts - 1}:0>'),
                    ('srq', f'srq<{num_insts - 1}:0>')
                ]
            else:
                conn_list = [
                    ('in_p', f'a<{num_insts - 2}:0>,dlyin'),
                    ('bk', f'bk<{num_insts - 1}:0>'),
                    ('ci_p', f'b{num_insts - 1},b<{num_insts - 2}:0>'),
                    ('out_p', f'b<{num_insts - 2}:0>,dlyout'),
                    ('co_p', f'a{num_insts - 1},a<{num_insts - 2}:0>'),
                    ('si', f'so<{num_insts - 2}:0>,iSI'),
                    ('so', f'SOOUT,so<{num_insts - 2}:0>'),
                ]

            if flop_char:
                conn_list.append(('bk1', f'flop_q<{num_insts - 1}:0>'))
                self.add_pin(f'flop_q<{num_insts - 1}:0>', TermType.output)
            self.rename_instance('XCELL', f'XCELL<{num_insts - 1}:0>', conn_list)
        elif num_insts == 2:
            conn_list = [
                ('in_p', f'a,dlyin'),
                ('bk', f'bk<{num_insts - 1}:0>'),
                ('ci_p', f'b{num_insts - 1},b'),
                ('out_p', 'b,dlyout'),
                ('co_p', f'a{num_insts - 1},a'),
                ('si', f'so,iSI'),
                ('so', f'SOOUT,so'),
            ]

            if output_sr_pins:
                conn_list += [('srqb', f'srqb<1:0>'), ('srq', f'srq<1:0>')]

            if flop_char:
                conn_list.append(('bk1', f'flop_q<{num_insts - 1}:0>'))
                self.add_pin(f'flop_q<{num_insts - 1}:0>', TermType.output)
            self.rename_instance('XCELL', f'XCELL<{num_insts - 1}:0>', conn_list)
        elif num_insts == 1:
            if flop_char:
                self.reconnect_instance_terminal('XCELL', 'bk1', 'flop_q<0>')
                self.add_pin('flop_q<0>', TermType.output)
            if output_sr_pins:
                self.reconnect_instance_terminal('XCELL', 'srq', 'srq')
                self.reconnect_instance_terminal('XCELL', 'srqb', 'srqb')

        else:
            raise ValueError(f'num_insts={num_insts} should be greater than 0.')

        if num_dum > 0:
            dc_core_params = dlycell_params['dc_core_params'].copy(remove=['output_sr_pins'])
            dum_params = dlycell_params.copy(remove=['flop_char', 'output_sr_pins'],
                                             append={'is_dum': True,
                                                     'dc_core_params': dc_core_params})
            self.instances['XDUM'].design(**dum_params)
            if num_dum > 1:
                suffix = f'<{num_dum - 1}:0>'
                conn_list = [
                    ('out_p', 'NC_out' + suffix),
                    ('co_p', 'NC_co' + suffix),
                    ('so', 'NC_so' + suffix),
                ]
                self.rename_instance('XDUM', 'XDUM' + suffix, conn_list)
                if flop:
                    self.rename_instance('XNC_so', 'XNC_so' + suffix,
                                         [('noConn', 'NC_so' + suffix)])
                else:
                    self.remove_instance('XNC_so')
                self.rename_instance('XNC_co', 'XNC_co' + suffix, [('noConn', 'NC_co' + suffix)])
                self.rename_instance('XNC_out', 'XNC_out' + suffix, [('noConn', 'NC_out' + suffix)])
        else:
            for inst in ['XDUM', 'XNC_so', 'XNC_co', 'XNC_out']:
                self.remove_instance(inst)

        if output_sr_pins:
            if num_insts == 2:
                raise ValueError('oops not supported')
            pin_name_list = [
                ('bk', f'bk<{num_insts - 1}:0>'),
                ('b', f'b<{num_insts - 1}:0>'),
                ('a', f'a<{num_insts - 1}:0>'),
            ] if num_insts > 1 else []
            self.add_pin(f'srq<{num_insts-1}:0>', TermType.output)
            self.add_pin(f'srqb<{num_insts-1}:0>', TermType.output)
        else:
            pin_name_list = [
                ('bk', f'bk<{num_insts - 1}:0>'),
                ('b', f'b{num_insts - 1}'),
                ('a', f'a{num_insts - 1}'),
            ] if num_insts > 1 else []

        for old_name, new_name in pin_name_list:
            self.rename_pin(old_name, new_name)

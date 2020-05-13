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

from typing import Mapping, Any, Tuple, Dict, Optional

from bag.simulation.cache import DesignInstance


def setup_digital_tran(specs: Mapping[str, Any], dut: Optional[DesignInstance],
                       wrapper_params: Optional[Mapping[str, Any]] = None,
                       **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Setup DigitalTranTB or its subclasses.

    This method handles connecting supplies/resets and hooking up pin connections
    for generic blocks.
    """
    tbm_specs_orig: Mapping[str, Any] = specs['tbm_specs']

    if wrapper_params is None:
        # check if it is specified in specs
        wrapper_params = specs.get('wrapper_params', None)
    if wrapper_params is None:
        dut_pins = list(dut.sch_master.pins.keys())
        pwr_domain: Mapping[str, Tuple[str, str]] = tbm_specs_orig['pwr_domain']
        tb_params = {}
    else:
        dut_pins = wrapper_params['pins']
        pwr_domain: Mapping[str, Tuple[str, str]] = wrapper_params['pwr_domain']
        tb_params = dict(dut_lib=wrapper_params['lib'], dut_cell=wrapper_params['cell'],
                         dut_params=wrapper_params['params'])

    tbm_specs = dict(**tbm_specs_orig)
    tbm_specs['pwr_domain'] = pwr_domain
    tbm_specs['dut_pins'] = dut_pins
    tbm_specs.update(kwargs)

    return tbm_specs, tb_params

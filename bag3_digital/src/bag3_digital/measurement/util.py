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

from typing import Mapping, Any, Tuple, Sequence, Dict, Iterable, Optional

from pybag.enum import TermType
from pybag.core import get_cdba_name_bits

from bag.simulation.cache import DesignInstance

from bag3_liberty.util import cdba_to_unusal

from bag3_testbenches.measurement.tran.digital import DigitalTranTB


def get_in_buffer_pin_names(pin: str) -> Tuple[str, str]:
    base = cdba_to_unusal(pin)
    return f'{base}_m_', f'{base}_dut_'


def get_digital_wrapper_params(specs: Mapping[str, Any], dut: DesignInstance,
                               in_pins: Iterable[str],
                               buf_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Setup digital wrapper in CombLogicTimingTB.

    This method handles connecting supplies/resets, adding in input buffers, and hooking up
    pin connections for generic blocks.
    """
    tbm_specs_orig: Mapping[str, Any] = specs['tbm_specs']

    if buf_params is None:
        buf_params = specs['buf_params']

    pwr_domain_orig: Mapping[str, Tuple[str, str]] = tbm_specs_orig['pwr_domain']
    diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]] = tbm_specs_orig.get('diff_list', [])

    diff_lookup = DigitalTranTB.get_diff_lookup(diff_list)

    # first, gather DUT pins
    wrap_conns = {}
    wrap_in_list = []
    wrap_out_list = []
    wrap_sup_set = set()
    dut_bit_conns = {}
    bit_list_info = []
    for pin, term_type in dut.sch_master.pins.items():
        bit_list = get_cdba_name_bits(pin)
        bit_list_info.append((pin, bit_list))
        for bit_name in bit_list:
            dut_bit_conns[bit_name] = bit_name
        wrap_conns[term_type] = term_type
        if term_type is TermType.input:
            wrap_in_list.append(pin)
        elif term_type is TermType.output:
            wrap_out_list.append(pin)
        else:
            wrap_sup_set.add(pin)

    # get all input pins that need buffers
    buf_pins = set()
    for pin in in_pins:
        buf_pins.add(pin)

        diff_grp = diff_lookup.get(pin, None)
        if diff_grp is not None:
            buf_pins.update(diff_grp[0])
            buf_pins.update(diff_grp[1])

    # create buffers
    buf_list = []
    pwr_domain = dict(**pwr_domain_orig)
    for pin in buf_pins:
        pwr_tup = DigitalTranTB.get_pin_supplies(pin, pwr_domain)
        buf_vss_name, buf_vdd_name = pwr_tup
        buf_mid, buf_out = get_in_buffer_pin_names(pin)
        pwr_domain[buf_mid] = pwr_domain[buf_out] = pwr_tup
        buf_list.append((buf_params, pin, dict(out=buf_out, mid=buf_mid, VDD=buf_vdd_name,
                                               VSS=buf_vss_name)))
        dut_bit_conns[pin] = buf_out
        wrap_conns[buf_mid] = buf_mid
        wrap_conns[buf_out] = buf_out
        wrap_out_list.append(buf_mid)
        wrap_out_list.append(buf_out)
        # NOTE: sometimes input supplies are not one of the power pins
        wrap_sup_set.add(buf_vss_name)
        wrap_sup_set.add(buf_vdd_name)

    # get dut_conns from dut_bit_conns
    dut_conns = {}
    for pin, bit_list in bit_list_info:
        net_list = [dut_bit_conns.get(bit_name, bit_name) for bit_name in bit_list]
        dut_conns[pin] = ','.join(net_list)

    # get wrapper params
    wrap_pins = wrap_in_list + wrap_out_list
    wrap_pins.extend(wrap_sup_set)
    wrapper_params = dict(
        lib='bag3_digital',
        cell='digital_db_top',
        params=dict(
            dut_conns=dut_conns,
            buf_params=buf_list,
            in_pin_list=wrap_in_list,
            out_pin_list=wrap_out_list,
            sup_pin_list=list(wrap_sup_set),
        ),
        pins=wrap_pins,
        pwr_domain=pwr_domain,
    )
    return wrapper_params

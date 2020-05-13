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

from __future__ import annotations

from typing import Any, Optional, Sequence, Mapping, Union, Type, cast

import pprint

from numpy.polynomial import polynomial
import matplotlib.pyplot as plt

from bag.layout.template import TemplateBase
from bag.simulation.design import DesignerBase

from ..digital.timing import CombLogicTimingTB


class FanoutPlotter(DesignerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def async_design(self, impl_cell: str, lay_class: Union[Type[TemplateBase], str],
                           dut_params: Mapping[str, Any], dut_conns: Mapping[str, Any],
                           sim_envs: Sequence[str], vdd: float, trf: float, tbit: float,
                           c_start: float, c_stop: float, num: int, out_invert: bool,
                           tran_options: Optional[Mapping[str, Any]] = None
                           ) -> Mapping[str, Any]:
        if len(sim_envs) != 1:
            self.error('Right now only support single corner.')

        tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns=dut_conns
        )
        tbm_specs = dict(
            sim_envs=sim_envs,
            thres_lo=0.1,
            thres_hi=0.9,
            stimuli_pwr='vdd',
            tstep=None,
            sim_params=dict(
                vdd=vdd,
                cload=c_start,
                tbit=tbit,
                trf=trf,
            ),
            save_outputs=['in', 'out'],
            rtol=1e-8,
            atol=1e-22,
            tran_options=tran_options,
            swp_info=[('cload', dict(type='LINEAR', start=c_start, stop=c_stop, num=num))],
        )
        self.log('Creating DUT')
        dut = await self.async_new_dut(impl_cell, lay_class, dut_params)
        self.log('Creating TestbenchManager')
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))
        self.log('Running simulation')
        sim_result = await self.async_simulate_tbm_obj('fanout', dut, tbm, tb_params)
        data = sim_result.data

        cvec = data['cload']
        tr, tf = tbm.calc_output_delay(data, 'in', 'out', out_invert)
        # remove corner axis
        tr = tr[0, :]
        tf = tf[0, :]

        tr_poly, rfit_info = polynomial.polyfit(cvec, tr, 1, full=True)
        tf_poly, ffit_info = polynomial.polyfit(cvec, tf, 1, full=True)

        tr_fit = cvec * tr_poly[1] + tr_poly[0]
        tf_fit = cvec * tf_poly[1] + tf_poly[0]

        ans = dict(
            rise=dict(
                m=tr_poly[1],
                y=tr_poly[0],
                resid=rfit_info[0],
                rcond=rfit_info[3],
            ),
            fall=dict(
                m=tf_poly[1],
                y=tf_poly[0],
                resid=ffit_info[0],
                rcond=ffit_info[3],
            ),
        )

        pprint.pprint(ans)

        self.log('Plotting graphs')
        plt.figure(1)
        plt.plot(cvec, tr, 'go', label='tr')
        plt.plot(cvec, tr_fit, '-b', label='tr_fit')
        plt.plot(cvec, tf, 'ro', label='tf')
        plt.plot(cvec, tf_fit, '-k', label='tf_fit')
        plt.legend()
        plt.show()

        return ans

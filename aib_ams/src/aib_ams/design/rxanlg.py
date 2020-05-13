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

from typing import Dict, Any, Mapping, Optional, Type

import math

from bag.layout.template import TemplateBase
from bag.io.file import write_yaml

from bag3_digital.design.base import DigitalDesigner
from bag3_digital.design.lvl_shift_de import LvlShiftDEDesigner
from bag3_digital.design.lvl_shift_ctrl import LvlShiftCtrlDesigner

from .se_to_diff_en import SingleToDiffEnableDesigner
from ..layout.rxanlg import RXAnalog


class RXAnalogDesigner(DigitalDesigner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:

        self._por_lv_specs: Optional[Dict[str, Any]] = None
        self._data_lv_specs: Optional[Dict[str, Any]] = None
        self._se_en_specs: Optional[Dict[str, Any]] = None
        self._ctrl_lv_specs: Optional[Dict[str, Any]] = None

        super().__init__(*args, **kwargs)

    @classmethod
    def get_dut_lay_class(cls) -> Optional[Type[TemplateBase]]:
        return RXAnalog

    def commit(self) -> None:
        """Updates object design parameters
        """
        super().commit()

        specs = self.dsn_specs
        sup_values: Mapping[str, Any] = specs['sup_values']
        data_lv_results: Optional[Mapping[str, Any]] = specs.get('data_lv_results', None)
        se_en_results: Optional[Mapping[str, Any]] = specs.get('se_en_results', None)
        ctrl_lv_results: Optional[Mapping[str, Any]] = specs.get('ctrl_lv_results', None)
        por_lv_results: Optional[Mapping[str, Any]] = specs.get('por_lv_results', None)

        if data_lv_results is None:
            sup_values_data_lv = dict(
                VDDI=sup_values['VDDIO'], VDD=sup_values['VDDCore'], VSS=sup_values['VSS']
            )
            self._data_lv_specs = dict(**specs['data_lv_specs'])
            for key in ['c_load', 'w_min', 'w_res', 'ridx_n', 'ridx_p', 'c_unit_p',
                        'c_unit_n', 'search_step', 'buf_config', 'search_params', 'tile_name',
                        'tile_specs', 'dig_tran_specs']:
                self._data_lv_specs[key] = specs[key]
            self._data_lv_specs['sup_values'] = sup_values_data_lv
            self._data_lv_specs['lv_params'] = lv_params = dict(**self._data_lv_specs['lv_params'])
            lv_params['has_rst'] = lv_params['in_upper'] = lv_params['dual_output'] = True
            lv_params['stack_p'] = 2
        else:
            self._data_lv_specs = None

        if se_en_results is None:
            sup_values_se_en = dict(VDD=sup_values['VDDIO'], VSS=sup_values['VSS'])
            self._se_en_specs = dict(**specs['se_en_specs'])
            for key in ['w_min', 'w_res', 'ridx_n', 'ridx_p', 'c_unit_p',
                        'c_unit_n', 'max_iter', 'search_step', 'buf_config', 'search_params',
                        'sign_off_envs', 'tile_name', 'tile_specs', 'dig_tran_specs']:
                self._se_en_specs[key] = specs[key]
            self._se_en_specs['sup_values'] = sup_values_se_en
        else:
            self._se_en_specs = None

        if ctrl_lv_results is None:
            sup_values_ctrl_lv = dict(
                VDDI=sup_values['VDDCore'], VDD=sup_values['VDDIO'], VSS=sup_values['VSS']
            )
            self._ctrl_lv_specs = dict(**specs['ctrl_lv_specs'])
            for key in ['buf_config', 'search_params', 'tile_name', 'tile_specs',
                        'dig_tran_specs']:
                self._ctrl_lv_specs[key] = specs[key]
            self._ctrl_lv_specs['sup_values'] = sup_values_ctrl_lv
            self._ctrl_lv_specs['has_rst'] = self._ctrl_lv_specs['is_ctrl'] = True
            self._ctrl_lv_specs['dual_output'] = True
        else:
            self._ctrl_lv_specs = None

        if por_lv_results is None:
            sup_values_por_lv = dict(
                VDDI=sup_values['VDDCore'], VDD=sup_values['VDDIO'], VSS=sup_values['VSS']
            )
            self._por_lv_specs = dict(**specs['por_lv_specs'])
            for key in ['w_min', 'w_res', 'ridx_n', 'ridx_p', 'c_unit_p',
                        'c_unit_n', 'search_step', 'buf_config', 'search_params', 'tile_name',
                        'tile_specs', 'dig_tran_specs']:
                self._por_lv_specs[key] = specs[key]
            self._por_lv_specs['sup_values'] = sup_values_por_lv
            self._por_lv_specs['lv_params'] = dict(
                has_rst=False,
                in_upper=True,
                dual_output=True,
                stack_p=1,
            )
        else:
            self._por_lv_specs = None

    async def async_design(self, **kwargs: Any) -> Mapping[str, Any]:
        """This function extracts design parameters and calls sub-hierarchy design functions.
        It passes parameters between the results of each sub-hierarchy design to accomplish
        logical-effort based design. If layout parameters are passed in through the design
        specs, they will be returned instead of running the design procedures.

        Passed in kwargs are the same as self.dsn_specs.

        Parameters
        ----------
        kwargs: Any
            data_lv_specs: Mapping[str, Any]
                Data Level Shifter design parameters
            se_en_specs: Mapping[str, Any]
                Single Ended to Differential design parameters
            ctrl_lv_specs: Mapping[str, Any]
                Control / Enable Level Shifters design parameters
            por_lv_specs: Mapping[str, Any]
                POR Level Shifter design parameters
            c_load: float
                Target load capacitance on odat
            c_odat_async: float
                Target load capacitance on odat_async
            c_por_vccl_tx: float
                Target load capacitance from the TX on por_vccl
            fanout_odat_async: float
                Target fanout for odat_async output inverter
            w_n_inv: Union[int, float]
                NMOS width for POR Level Shifter input buffer
            w_p_inv: Union[int, float]
                PMOS width for POR Level Shifter input buffer
            yaml_file: str
                Output file location to write designed generator parameters
            data_lv_results: Optional[Mapping[str, Any]]
                If provided, return these generator parameters instead of running the design
                procedures
            se_en_results: Optional[Mapping[str, Any]]
                If provided, return these generator parameters instead of running the design
                procedures
            ctrl_lv_results: Optional[Mapping[str, Any]]
                If provided, return these generator parameters instead of running the design
                procedures
            por_lv_results: Optional[Mapping[str, Any]]
                If provided, return these generator parameters instead of running the design
                procedures

            Below are global specs shared and passed to each of the designers
            w_min: Union[int, float]
                Minimum width
            w_res: Union[int, float]
                Width resolution
            c_unit_n: float
                Unit NMOS transistor capacitance for w=1, seg=1
            c_unit_p: float
                Unit PMOS transistor capacitance for w=1, seg=1
            dig_tran_specs: Mapping[str, Any]
                DigitalTranTB testbench specs
            search_params: Mapping[str, Any]
                Parameters used for capacitor size binary search
            search_step: int
                Binary search step size
            max_iter: int
                Maximum allowed iterations to search for converge in binary search
            buf_config: Mapping[str, Any]
                Buffer parameters, used in DigitalTranTB and capacitor size search
            sign_off_envs: Sequence[str]
                Corners used for sign off
            sup_values: Mapping[str, Any]
                Per-corner supply values
            tile_name: str
                Tile name for layout.
            tile_specs: Mapping[str, Any]
                Tile specifications for layout.
            ridx_n: int
                NMOS transistor row
            ridx_p: int
                PMOS transistor Row

        Returns
        -------
        ans: Mapping[str, Any]
            Design summary
        """
        specs = self.dsn_specs
        yaml_file: str = specs.get('yaml_file', '')

        data_lv_results = await self.design_data_lv()
        se_en_results = await self.design_se_en(data_lv_results['c_in'])

        c_en = max(se_en_results['c_en_se'], se_en_results['c_en_match'])
        ctrl_lv_results = await self.design_ctrl_lv(c_en)

        c_por_vccl = max(ctrl_lv_results['c_rst_out'], ctrl_lv_results['c_rst_casc']) * 2
        c_por_vccl += self.dsn_specs['c_por_vccl_tx']

        por_lv_results = await self.design_por_lv(c_por_vccl)

        c_por_core = max(data_lv_results['c_rst_out'], data_lv_results['c_rst_casc']) * 2
        c_por_core += por_lv_results['c_in']
        buf_por_lv_params = self.get_buf_params(c_por_core)

        inv_params = self.get_inv_params(self.dsn_specs['c_odat_async'],
                                         self.dsn_specs['fanout_odat_async'])

        se_params = se_en_results['se_params']
        match_params = se_en_results['match_params']
        data_lv_params = data_lv_results['lv_params']
        por_lv_params = por_lv_results['lv_params']

        for table in [se_params, match_params, data_lv_params, por_lv_params]:
            table.pop('pinfo', None)
            table.pop('ridx_n', None)
            table.pop('ridx_p', None)

        rx_params = dict(
            se_params=se_params,
            match_params=match_params,
            inv_params=inv_params,
            data_lv_params=data_lv_params,
            ctrl_lv_params=ctrl_lv_results['dut_params']['lv_params'],
            por_lv_params=por_lv_params,
            buf_ctrl_lv_params=ctrl_lv_results['dut_params']['in_buf_params'],
            buf_por_lv_params=buf_por_lv_params,
        )
        ans = dict(
            rx_params=rx_params,
        )

        if yaml_file:
            write_yaml(self.work_dir / yaml_file, ans)

        return ans

    async def design_data_lv(self) -> Mapping[str, Any]:
        if self._data_lv_specs is None:
            return self.dsn_specs['data_lv_results']

        dsn = self.new_designer(LvlShiftDEDesigner, self._data_lv_specs)
        return await dsn.async_design()

    async def design_se_en(self, c_load: float) -> Mapping[str, Any]:
        if self._se_en_specs is None:
            return self.dsn_specs['se_en_results']

        self._se_en_specs['c_load'] = c_load
        dsn = self.new_designer(SingleToDiffEnableDesigner, self._se_en_specs)
        return await dsn.async_design()

    async def design_ctrl_lv(self, c_load: float) -> Mapping[str, Any]:
        if self._ctrl_lv_specs is None:
            return self.dsn_specs['ctrl_lv_results']

        self._ctrl_lv_specs['cload'] = c_load
        dsn = self.new_designer(LvlShiftCtrlDesigner, self._ctrl_lv_specs)
        return await dsn.async_design(**dsn.dsn_specs)

    async def design_por_lv(self, c_load: float) -> Mapping[str, Any]:
        if self._por_lv_specs is None:
            return self.dsn_specs['por_lv_results']

        self._por_lv_specs['c_load'] = c_load
        dsn = self.new_designer(LvlShiftDEDesigner, self._por_lv_specs)
        return await dsn.async_design()

    def get_buf_params(self, c_load: float) -> Mapping[str, Any]:
        specs = self.dsn_specs
        c_unit_p: float = specs['c_unit_p']
        c_unit_n: float = specs['c_unit_n']

        if self._por_lv_specs is not None:
            fanout = self._por_lv_specs['fanout_inv']
        else:
            fanout = min(10.0, specs['fanout_odat_async'] * 2)

        inv1_params = self.get_inv_params(c_load, fanout)
        c_inv1 = (c_unit_p * inv1_params['seg_p'] * inv1_params['w_p'] +
                  c_unit_n * inv1_params['seg_n'] * inv1_params['w_n'])
        inv0_params = self.get_inv_params(c_inv1 + c_load, fanout)
        return dict(
            segp_list=[inv0_params['seg_p'], inv1_params['seg_p']],
            segn_list=[inv0_params['seg_n'], inv1_params['seg_n']],
            w_p=[inv0_params['w_p'], inv1_params['w_p']],
            w_n=[inv0_params['w_n'], inv1_params['w_n']],
        )

    def get_inv_params(self, c_load: float, fanout: float) -> Mapping[str, Any]:
        specs = self.dsn_specs
        c_unit_p: float = specs['c_unit_p']
        c_unit_n: float = specs['c_unit_n']
        w_p: int = specs['w_p_inv']
        w_n: int = specs['w_n_inv']

        c_unit = c_unit_p * w_p + c_unit_n
        seg = max(1, int(math.ceil(c_load / c_unit / fanout)))
        return dict(seg_p=seg, seg_n=seg, w_p=w_p, w_n=w_n)

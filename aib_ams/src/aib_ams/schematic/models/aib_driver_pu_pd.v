// SPDX-License-Identifier: Apache-2.0
// Copyright 2020 Blue Cheetah Analog Design Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

{{ _header }}

    logic out_temp;

    always_comb begin
        // puenb connects to PMOS, pden connects to NMOS
        casez ({VDD, VSS, puenb, pden})
           4'b10_00: out_temp = 1'b1;
           4'b10_01: out_temp = 1'bx;
           4'b10_10: out_temp = 1'bz;
           4'b10_11: out_temp = 1'b0;
           4'b00_??: out_temp = 1'b0;
           default:  out_temp = 1'bx;
        endcase
    end

    assign{% if not _sch_params['strong'] %} (weak0, weak1){% endif %} out = out_temp;

endmodule

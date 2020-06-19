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
{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};
logic temp;

{% if _sch_params['has_rsthb'] %}
always_comb begin
    casez ({en, enb, rsthb, VDD, VSS})
        5'b??010: temp = 1'b1;
        5'b10110: temp = ~in;
        5'b01110: temp = 1'bz;
        5'b11110: temp = in ? 1'b0 : 1'bz;
        5'b00110: temp = in ? 1'bz : 1'b1;
        5'b???00: temp = 1'b0;
        default : temp = 1'bx;
    endcase
end
{% else %}
always_comb begin
    casez ({en, enb, VDD, VSS})
        4'b1010: temp = ~in;
        4'b0110: temp = 1'bz;
        4'b1110: temp = in ? 1'b0 : 1'bz;
        4'b0010: temp = in ? 1'bz : 1'b1;
        4'b??00: temp = 1'b0;
        default : temp = 1'bx;
    endcase
end
{% endif %}

assign #DELAY out = temp;

endmodule

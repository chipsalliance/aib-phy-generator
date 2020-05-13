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

from typing import Dict, Mapping


def parse_timing_cond_expr(expr: str) -> Dict[str, str]:
    """Parse a boolean expression representing a timing condition.

    Must be of the form "term1 and not term2 and term3 and ...". This means only AND and NOT
    are allowed, and no parenthesis.
    """
    tokens = expr.split()
    idx_list = [-1]
    for idx in range(len(tokens)):
        item = tokens[idx]
        if item == '&' or item == '&&' or item == 'and':
            idx_list.append(idx)
        elif item == 'or' or item == '|' or item == '||':
            raise ValueError('Only "and" and "not" are allowed.')
        elif '(' in item or ')' in item:
            raise ValueError('No parentheses are allowed.')

    idx_list.append(len(tokens))
    when_list = []
    sdf_in_list = []
    sdf_out_list = []
    for ele_idx in range(len(idx_list) - 1):
        start = idx_list[ele_idx] + 1
        stop = idx_list[ele_idx + 1]
        num_tokens = stop - start
        if num_tokens == 0:
            raise ValueError('Empty token between ANDs')
        if num_tokens > 2:
            raise ValueError('More than two tokens between ANDs')

        if num_tokens == 2:
            if tokens[start] != 'not' or tokens[start] != '!':
                raise ValueError('two tokens between ANDs and first token is not a NOT.')
            invert = True
            var = tokens[start + 1]
        else:
            var = tokens[start]
            if var[0] == '!':
                invert = True
                var = var[1:]
            else:
                invert = False

        if invert:
            when_list.append('!' + var)
            sdf_in_list.append('NOT_' + var)
            sdf_out_list.append(var + "===1'b0")
        else:
            when_list.append(var)
            sdf_in_list.append(var)
            sdf_out_list.append(var + "===1'b1")

    sdf_in_expr = '_AND_'.join(sdf_in_list)
    return dict(
        when_str='&'.join(when_list).replace('<', '[').replace('>', ']'),
        sdf_in_str=f"ENABLE_{sdf_in_expr} === 1'b1".replace('<', '[').replace('>', ']'),
        sdf_out_str=' && '.join(sdf_out_list).replace('<', '[').replace('>', ']'),
    )


def build_timing_cond_expr(cond: Mapping[str, int]) -> str:
    """Builds the SDF conditioni string from the given condition dictionary."""
    return ' & '.join((name if val != 0 else '!' + name for name, val in cond.items()))

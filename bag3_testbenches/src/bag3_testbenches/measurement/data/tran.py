# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

"""Transient simulation related data processing methods."""
from __future__ import annotations

from typing import Sequence, List, Tuple, Any, Iterable, Optional, Callable, Union

from enum import Flag, auto
from itertools import islice

import numpy as np
from scipy.interpolate import interp1d

from bag.util.search import BinaryIterator


class EdgeType(Flag):
    RISE = auto()
    FALL = auto()
    CROSS = RISE | FALL

    @property
    def opposite(self) -> EdgeType:
        if self is EdgeType.RISE:
            return EdgeType.FALL
        if self is EdgeType.FALL:
            return EdgeType.RISE
        if self is EdgeType.CROSS:
            return EdgeType.CROSS
        raise ValueError(f'Unknown edge type: {self.name}')


def interp1d_no_nan(tvec: np.ndarray, yvec: np.ndarray
                    ) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    tsize = len(tvec)
    if np.isnan(tvec[-1]):
        bin_iter = BinaryIterator(1, tsize + 1)
        while bin_iter.has_next():
            delta = bin_iter.get_next()
            if np.isnan(tvec[tsize - delta]):
                bin_iter.save()
                bin_iter.up()
            else:
                bin_iter.down()
        tsize -= bin_iter.get_last_save()

    return interp1d(tvec[:tsize], yvec[:tsize], assume_sorted=True, copy=False)


def bits_to_pwl_iter(values: Sequence[Any]) -> Iterable[Tuple[float, float, float, Any]]:
    """Convert discrete samples to PWL waveform.

    This method yields coefficients to td, tbit and trf, so user can generate symbolic PWL
    waveform files.  Note that td must be positive.

    Parameters
    ----------
    values : List[float]
        list of values for each bit.

    Yields
    ------
    td_scale : float
        coefficient for td
    tbit_scale : float
        coefficient for tbit
    trf : float
        coefficient for trf
    val : Any
        the value
    """
    cur_info = [1, 0, 0, values[0]]
    yield tuple(cur_info)
    cur_info[1] += 1
    cur_info[2] -= 0.5
    for ycur in islice(values, 1, None):
        if ycur != cur_info[3]:
            yield tuple(cur_info)
            cur_info[3] = ycur
            cur_info[2] += 1
            yield tuple(cur_info)
            cur_info[1] += 1
            cur_info[2] -= 1
        else:
            cur_info[1] += 1

    # output last point
    yield tuple(cur_info)


def get_first_crossings(tvec: np.ndarray, yvec: np.ndarray, threshold: Union[float, np.ndarray],
                        start: Union[float, np.ndarray] = 0,
                        stop: Union[float, np.ndarray] = float('inf'),
                        etype: EdgeType = EdgeType.CROSS, rtol: float = 1e-8, atol: float = 1e-22,
                        shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Find the first time where waveform crosses a given threshold.

    tvec and yvec can be multi-dimensional, in which case the waveforms are stored in the
    last axis.  The returned numpy array will have the same shape as yvec with the last
    axis removed.  If the waveform never crosses the threshold, positive infinity will be
    returned.
    """
    swp_shape = yvec.shape[:-1]
    if shape is None:
        shape = yvec.shape[:-1]

    try:
        th_vec = np.broadcast_to(np.asarray(threshold), swp_shape)
        start = np.broadcast_to(np.asarray(start), swp_shape)
        stop = np.broadcast_to(np.asarray(stop), swp_shape)
    except ValueError as err:
        raise ValueError('Failed to make threshold/start/stop the same shape as data.  '
                         'Make sure they are either scalar or has the same sweep shape.') from err

    t_shape = tvec.shape
    nlast = t_shape[len(t_shape) - 1]

    yvec = yvec.reshape(-1, nlast)
    tvec = tvec.reshape(-1, nlast)
    th_vec = th_vec.flatten()
    t0_vec = start.flatten()
    t1_vec = stop.flatten()
    n_swp = th_vec.size
    ans = np.empty(n_swp)
    num_tvec = tvec.shape[0]

    for idx in range(n_swp):
        cur_thres = th_vec[idx]
        cur_t0 = t0_vec[idx]
        cur_t1 = t1_vec[idx]
        ans[idx] = _get_first_crossings_time_1d(tvec[idx % num_tvec, :], yvec[idx, :], cur_thres,
                                                cur_t0, cur_t1, etype, rtol, atol)
    return ans.reshape(shape)


def _get_first_crossings_time_1d(tvec: np.ndarray, yvec: np.ndarray, threshold: float,
                                 start: float, stop: float, etype: EdgeType, rtol: float,
                                 atol: float) -> float:
    # eliminate NaN from time vector in cases where simulation time is different between runs.
    mask = ~np.isnan(tvec)
    tvec = tvec[mask]
    yvec = yvec[mask]

    sidx = np.searchsorted(tvec, start)
    eidx = np.searchsorted(tvec, stop)
    if eidx < tvec.size and np.isclose(stop, tvec[eidx], rtol=rtol, atol=atol):
        eidx += 1

    # quantize waveform values, then detect edge.
    dvec = np.diff((yvec[sidx:eidx] >= threshold).astype(int))

    ans = float('inf')
    if EdgeType.RISE in etype:
        sel_mask = np.maximum(dvec, 0)
        arg = sel_mask.argmax()
        if arg != 0 or sel_mask[0] != 0:
            # has edge
            ans = _get_first_crossings_helper(tvec, yvec, threshold, sidx, arg)
    if EdgeType.FALL in etype:
        sel_mask = np.minimum(dvec, 0)
        arg = sel_mask.argmin()
        if arg == 0 and sel_mask[0] == 0:
            # no edge
            return ans
        return min(ans, _get_first_crossings_helper(tvec, yvec, threshold, sidx, arg))
    return ans


def _get_first_crossings_helper(tvec: np.ndarray, yvec: np.ndarray,
                                threshold: float, idx0: int, arg: int) -> float:
    arg += idx0
    t0 = tvec[arg]
    y0 = yvec[arg]
    t1 = tvec[arg + 1]
    y1 = yvec[arg + 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        ans = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
    return ans if t0 <= ans <= t1 else np.inf

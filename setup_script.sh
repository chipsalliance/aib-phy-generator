#!/bin/bash

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

git clone https://github.com/bluecheetah/xbase.git xbase_bcad
git clone https://github.com/bluecheetah/bag.git BAG_framework
git clone https://github.com/bluecheetah/cds_ff_mpt.git cds_ff_mpt

mkdir -p BAG_framework/pybag/_build/lib
cd BAG_framework/pybag/_build/lib
ln -s ../../../../bag3d0_rhel60_64/pybag .
cd ../../../../


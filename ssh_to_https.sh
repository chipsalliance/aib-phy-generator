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

change_gitmodules() {
	perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules
}

full_update() {

	change_gitmodules

	git submodule sync
	git submodule update --init
}

full_update

folder_list=("BAG_framework" "bag3_testbenches" "BAG_framework/pybag" "BAG_framework/pybag/cbag" "BAG_framework/pybag/pybind11_generics" "BAG_framework/pybag/pybind11_generics/pybind11")


for folder in ${folder_list[*]};
do
echo $folder
cd $folder; full_update; cd -
done

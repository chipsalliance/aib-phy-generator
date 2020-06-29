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

if [ -z ${CDS_INST_DIR} ]
then
    echo "CDS_INST_DIR is unset"
    exit 1
fi


# setup symlink to compiled pybag
mkdir -p BAG_framework/pybag/_build/lib
cd BAG_framework/pybag/_build/lib
ln -s ../../../../bag3d0_rhel60_64/pybag .
cd ../../../../

# setup symlink for files
ln -s cds_ff_mpt/workspace_setup/bag_config.yaml bag_config.yaml
ln -s cds_ff_mpt/workspace_setup/.cdsenv .cdsenv
ln -s cds_ff_mpt/workspace_setup/.cdsinit .cdsinit
ln -s cds_ff_mpt/workspace_setup/cds.lib.core cds.lib.core
ln -s cds_ff_mpt/workspace_setup/display.drf display.drf
ln -s cds_ff_mpt/workspace_setup/.gitignore .gitignore
ln -s cds_ff_mpt/workspace_setup/leBindKeys.il leBindKeys.il
ln -s cds_ff_mpt/workspace_setup/pvtech.lib pvtech.lib
ln -s BAG_framework/run_scripts/start_bag_ICADV12d3.il start_bag.il
ln -s BAG_framework/run_scripts/virt_server.sh virt_server.sh
ln -s BAG_framework/run_scripts/run_bag.sh run_bag.sh

# setup cadence shared library linkage
mkdir cadence_libs

declare -a lib_arr=("libblosc.so"
                    "libblosc.so.1"
                    "libblosc.so.1.11.4"
                    "libcdsCommon_sh.so"
                    "libcdsenvutil.so"
                    "libcdsenvxml.so"
                    "libcla_sh.so"
                    "libcls_sh.so"
                    "libdataReg_sh.so"
                    "libddbase_sh.so"
                    "libdrlLog.so"
                    "libfastt_sh.so"
                    "libgcc_s.so"
                    "libgcc_s.so.1"
                    "liblz4.so"
                    "liblz4.so.1"
                    "liblz4.so.1.7.1"
                    "libnffr.so"
                    "libnmp_sh.so"
                    "libnsys.so"
                    "libpsf.so"
                    "libsrr_fsdb.so"
                    "libsrr.so"
                    "libstdc++.so"
                    "libstdc++.so.5"
                    "libstdc++.so.5.0.7"
                    "libstdc++.so.6"
                    "libstdc++.so.6.0.22"
                    "libvirtuos_sh.so"
                    "libz_sh.so"
                   )

for libname in "${lib_arr[@]}"; do
    fpath=${CDS_INST_DIR}/tools/lib/64bit/${libname}
    if [ ! -f "$fpath" ]; then
        echo "WARNING: Cannot find packaged Virtuoso shared library ${fpath}; symlink will be broken."
    fi
    ln -s ${fpath} cadence_libs/${libname}
done

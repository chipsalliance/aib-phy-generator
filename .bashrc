#! /usr/bin/env bash
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
# clear out PYTHONPATH
export PYTHONPATH=""

### Setup BAG
source .bashrc_bag

# location of various tools
#export CDS_INST_DIR=/tools/cadence/ICADVM181
export CDS_INST_DIR=/p/psg/eda/cadence/virtuoso/ICADVM18.1.20191025.ISR7/linux64
#export PEGASUS_HOME=/tools/cadence/PEGASUS184
export PEGASUS_HOME=/nfs/site/disks/psg_eda_1/cadence/pvs/16.11.000/linux64
#export SRR_HOME=/tools/cadence/SRR
#export SPECTRE_HOME=/tools/cadence/SPECTRE181
export SPECTRE_HOME=/nfs/site/disks/psg_eda_1/cadence/spectre/18.10.235/linux64

#export CDSLIB_HOME=/tools/bag3/programs/cdsLibPlugin

export CDSHOME=$CDS_INST_DIR
#export CDSLIB_TOOL=${CDSLIB_HOME}/tools.lnx86
export MMSIM_HOME=${SPECTRE_HOME}

export OA_BIT=64

# PATH setup
#export PATH=${CDSLIB_TOOL}/bin:${PATH:-}
export PATH=${PEGASUS_HOME}/bin:${PATH}
export PATH=${CDS_INST_DIR}/tools/plot/bin:${PATH}
export PATH=${CDS_INST_DIR}/tools/dfII/bin:${PATH}
export PATH=${CDS_INST_DIR}/tools/bin:${PATH}
export PATH=${MMSIM_HOME}/bin:${PATH}
export PATH=${BAG_TOOLS_ROOT}/bin:${PATH}

# LD_LIBRARY_PATH setup
#export LD_LIBRARY_PATH=${CDSLIB_TOOL}/lib/64bit:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CDS_INST_DIR}/tools/lib/64bit:${LD_LIBRARY_PATH:-}
#export LD_LIBRARY_PATH=${SRR_HOME}/tools/lib/64bit:${LD_LIBRARY_PATH:-}
#export LD_LIBRARY_PATH=${BAG_TOOLS_ROOT}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${BAG_TOOLS_ROOT}/lib:${LD_LIBRARY_PATH}

# Virtuoso options
export SPECTRE_DEFAULTS=-E
export CDS_Netlisting_Mode="Analog"
export CDS_AUTO_64BIT=ALL
#export CDS_LIC_FILE=5280@login1.bcanalog.com

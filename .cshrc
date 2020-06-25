#! /usr/bin/env tcsh

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
setenv PYTHONPATH ""

### Setup BAG
source .cshrc_bag

# location of various tools
setenv CDS_INST_DIR /tools/cadence/ICADVM181
setenv PEGASUS_HOME /tools/cadence/PEGASUS184
setenv SPECTRE_HOME /tools/cadence/SPECTRE181
setenv QRC_HOME /tools/cadence/EXT191
setenv OA_CDS_ROOT ${CDS_INST_DIR}/oa_v22.60.007

setenv CDSHOME $CDS_INST_DIR
setenv MMSIM_HOME ${SPECTRE_HOME}

setenv OA_BIT 64
if (! $?OA_PLUGIN_PATH) then
    setenv OA_PLUGIN_PATH ${OA_CDS_ROOT}/data/plugins
else
    setenv OA_PLUGIN_PATH ${OA_CDS_ROOT}/data/plugins:${OA_PLUGIN_PATH}
endif

# PATH setup
setenv PATH ${PEGASUS_HOME}/bin:${PATH}
setenv PATH ${CDS_INST_DIR}/tools/plot/bin:${PATH}
setenv PATH ${CDS_INST_DIR}/tools/dfII/bin:${PATH}
setenv PATH ${CDS_INST_DIR}/tools/bin:${PATH}
setenv PATH ${MMSIM_HOME}/bin:${PATH}
setenv PATH ${QRC_HOME}/bin:${PATH}
setenv PATH ${BAG_TOOLS_ROOT}/bin:${PATH}

# LD_LIBRARY_PATH setup
if (! $?LD_LIBRARY_PATH) then
    setenv LD_LIBRARY_PATH ${BAG_WORK_DIR}/cadence_libs
else
    setenv LD_LIBRARY_PATH ${BAG_WORK_DIR}/cadence_libs:${LD_LIBRARY_PATH}
endif
setenv LD_LIBRARY_PATH ${BAG_TOOLS_ROOT}/lib:${LD_LIBRARY_PATH}

# Virtuoso options
setenv SPECTRE_DEFAULTS -E
setenv CDS_Netlisting_Mode "Analog"
setenv CDS_AUTO_64BIT ALL

# License setup
setenv CDS_LIC_FILE 5280@login1.bcanalog.com

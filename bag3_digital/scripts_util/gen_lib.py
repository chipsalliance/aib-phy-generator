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

import sys
import argparse
from pathlib import Path

from bag.io import read_yaml
from bag.core import BagProject

from bag3_digital.measurement.liberty.io import generate_liberty


def _info(etype, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(etype, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(etype, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


sys.excepthook = _info


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate libert file from spec file.')
    parser.add_argument('specs', help='Cell specification yaml file name.')
    parser.add_argument('-f', '--fake', dest='fake', action='store_true', default=False,
                        help='generate fake liberty file.')
    parser.add_argument('-x', '--extract', dest='extract', action='store_true', default=False,
                        help='Run extracted simulation.')
    parser.add_argument('--force_extract', action='store_true', default=False,
                        help='Force RC extraction even if layout/schematic are unchanged')
    parser.add_argument('-s', '--force_sim', action='store_true', default=False,
                        help='Force simulation even if simulation netlist is unchanged')
    parser.add_argument('-c', '--gen_sch', action='store_true', default=False,
                        help='Generate testbench schematics for debugging.')
    parser.add_argument('-e', '--gen_all_env', action='store_true', default=False,
                        help='Generate liberty file for all defined environments.')
    parser.add_argument('-l', '--export_lay', action='store_true', default=False,
                        help='Use CAD tool to export GDS.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs_path = Path(args.specs)
    root_dir = specs_path.parent
    specs = read_yaml(specs_path)
    lib_config = read_yaml(root_dir / 'lib_config.yaml')
    sim_config = read_yaml(root_dir / 'sim_config.yaml')
    generate_liberty(prj, lib_config, sim_config, specs, fake=args.fake, extract=args.extract,
                     force_sim=args.force_sim, force_extract=args.force_extract,
                     gen_sch=args.gen_sch, gen_all_env=args.gen_all_env, export_lay=args.export_lay)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if '_prj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['_prj']

    run_main(_prj, _args)

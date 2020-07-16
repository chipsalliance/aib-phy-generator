Instructions for Generating AIB blocks


Developed by Blue Cheetah Analog Design, Inc.
info@bcanalog.com

For each block, there will be a gds (.gds), lef (.lef), lib (.lib),
netlist (.net or .cdl), model (.v or .sv) and a shell (.v). There will
also be log files and some extra files containing the data necessary to
generate these results, but this document only shows the final outputs.

**dcc_delay_cell:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dcc_delay_cell.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dcc_delay_cell_lib.yaml

Output files:

gen_outputs/ip_blocks/dcc_delay_cell/

Should have the files:

-   dcc_delay_cell.gds

-   dcc_delay_cell_shell.v

-   dcc_delay_cell_tt_25_0p900_0p800.lib

-   dcc_delay_cell.cdl

-   dcc_delay_cell.lef

-   dcc_delay_cell.sv

**dcc_delay_line:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dcc_delay_line.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dcc_delay_line_lib.yaml

Output files:

gen_outputs/ip_blocks/dcc_delay_line/

Should have the files:

-   dcc_delay_line.gds

-   dcc_delay_line_shell.v

-   dcc_delay_line.v

-   dcc_delay_line.cdl

-   dcc_delay_line.lef

-   dcc_delay_line_tt_25_0p900_0p800.lib

**dcc_helper:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dcc_helper.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dcc_helper_lib.yaml

Output files:

gen_outputs/ip_blocks/dcc_helper/

Should have the files:

-   dcc_helper.lef

-   dcc_helper_tt_25_0p900_0p800.lib

-   schematic.net

-   dcc_helper.gds

-   dcc_helper_shell.v

-   dcc_helper.v

**dcc_interpolator:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dcc_phase_interp.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dcc_phase_interp_lib.yaml

Output files:

gen_outputs/ip_blocks/dcc_interpolator/

Should have the files:

-   dcc_interpolator.gds

-   dcc_interpolator_shell.v

-   dcc_interpolator_tt_25_0p900_0p800.lib

-   dcc_interpolator.cdl

-   dcc_interpolator.lef

-   dcc_interpolator.sv

**dcc_phasedet:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dcc_phase_det.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dcc_phase_det_lib.yaml

Output files:

gen_outputs/ip_blocks/dcc_phasedet/

Should have the files:

-   dcc_phasedet.lef

-   dcc_phasedet.sv

-   dcc_phasedet.gds

-   dcc_phasedet_shell.v

-   dcc_phasedet_tt_25_0p900_0p800.lib

-   schematic.net

**DIFF_CLK_RCVR:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/DIFF_CLK_RCVR.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/DIFF_CLK_RCVR_lib.yaml

Output files:

gen_outputs/ip_blocks/DIFF_CLK_RCVR/

Should have the files:

-   DIFF_CLK_RCVR.gds

-   DIFF_CLK_RCVR_shell.v

-   DIFF_CLK_RCVR_tt_25_0p900_0p800.lib

-   DIFF_CLK_RCVR.cdl

-   DIFF_CLK_RCVR.lef

-   DIFF_CLK_RCVR.sv

**dll_delay_cell:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dll_delay_cell.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dll_delay_cell_lib.yaml

Output files:

gen_outputs/ip_blocks/dll_delay_cell/

Should have the files:

-   dll_delay_cell.gds

-   dll_delay_cell_shell.v

-   dll_delay_cell_tt_25_0p900_0p800.lib

-   dll_delay_cell.cdl

-   dll_delay_cell.lef

-   dll_delay_cell.sv

**dll_delay_line:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dll_delay_line.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dll_delay_line_lib.yaml

Output files:

gen_outputs/ip_blocks/dll_delay_line/

Should have the files:

-   dll_delay_line.gds

-   dll_delay_line_shell.v

-   dll_delay_line.v

-   dll_delay_line.cdl

-   dll_delay_line.lef

-   dll_delay_line_tt_25_0p900_0p800.lib

**dll_interpolator:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dll_phase_interp.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dll_phase_interp_lib.yaml

Output files:

gen_outputs/ip_blocks/dll_interpolator/

Should have the files:

-   dll_interpolator.gds

-   dll_interpolator_shell.v

-   dll_interpolator_tt_25_0p900_0p800.lib

-   dll_interpolator.cdl

-   dll_interpolator.lef

-   dll_interpolator.sv

**dll_phasedet:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/dll_phasedet.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/dll_phasedet_lib.yaml

Output files:

gen_outputs/ip_blocks/dll_phasedet/

Should have the files:

-   dll_phasedet.lef

-   dll_phasedet.sv

-   dll_phasedet.gds

-   dll_phasedet_shell.v

-   dll_phasedet_tt_25_0p900_0p800.lib

-   schematic.net

**frontend:**

Gen cell command:

./run_bag.sh BAG_framework/run_scripts/gen_cell.py
data/aib_ams/specs_ip/frontend.yaml -raw -mod -lef

Gen Lib command:

./run_bag.sh bag3_digital/scripts_util/gen_lib.py
data/aib_ams/specs_ip/frontend_lib.yaml

Output files:

gen_outputs/ip_blocks/frontend /

Should have the files:

-   frontend.gds

-   frontend_shell.v

-   frontend.v

-   frontend.cdl

-   frontend.lef

-   frontend_tt_25_0p900_0p800.lib

Design scripts:

A design script will run through a design procedure for the given block,
and if successful in execution will generate similar collateral to the
previous generation scripts. Please note that since we are reusing the
lib file generation from the gen cell commands, the lib file is
generated in the same location as it was previously.

All the commands will follow the format of:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/\*.yaml

With the exact full command and output files detailed below.

**DCC Delay Line:**

Yaml file: dcc_delay_line.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dcc_delay_line.yaml

Folder:

gen_outputs/dsn_delay_line_final

Generated collateral:

-   aib_delay_line.gds

-   aib_delay_line.cdl

-   aib_delay_line.lef

-   aib_delay_line_shell.v

Lib file location:

gen_outputs/ip_blocks/dcc_delay_line/dcc_delay_line_tt_25_0p900_0p800.lib

**DCC Helper:**

Yaml file: dcc_helper.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dcc_helper.yaml

Folder:

gen_outputs/aib_dcc_helper_final

Generated collateral:

-   aib_dcc_helper.gds

-   aib_dcc_helper.cdl

-   aib_dcc_helper.sv

-   aib_dcc_helper.lef

-   aib_dcc_helper_shell.v

Lib file location:

gen_outputs/ip_blocks/dcc_helper/dcc_helper_tt_25_0p900_0p800.lib

**DCC Phase Interp:**

Yaml file: dcc_phase_interp.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dcc_phase_interp.yaml

Folder:

gen_outputs/dcc_phase_interp_final

Generated collateral:

-   AA_PhaseInterpWithDelay.gds

-   AA_PhaseInterpWithDelay.cdl

-   AA_PhaseInterpWithDelay.sv

-   AA_PhaseInterpWithDelay.lef

-   AA_PhaseInterpWithDelay_shell.v

Lib file location:

gen_outputs/ip_blocks/dcc_interpolator/dcc_interpolator_tt_25_0p900_0p800.lib

DCC Phase Det:

Yaml file: dcc_phasedet.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dcc_phasedet.yaml

Folder:

gen_outputs/dsn_phasedet_final

Generated collateral:

-   aib_phasedet_final.gds

-   aib_phasedet_final.cdl

-   aib_phasedet_final.sv

-   aib_phasedet_final.lef

-   aib_phasedet_final_shell.v

Lib file location:

gen_outputs/ip_blocks/dcc_phasedet/dcc_phasedet_tt_25_0p900_0p800.lib

**DLL Delay Line:**

Yaml file: dll_delay_line.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dll_delay_line.yaml

Folder:

gen_outputs/dsn_delay_line_final

Generated collateral:

-   aib_delay_line.gds

-   aib_delay_line.cdl

-   aib_delay_line.lef

-   aib_delay_line_shell.v

Lib file location:

gen_outputs/ip_blocks/dll_delay_line/dll_delay_line_tt_25_0p900_0p800.lib

**DLL Phase Interp:**

Yaml file: dll_phase_interp.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dll_phase_interp.yaml

Folder:

gen_outputs/dll_phase_interp_final

Generated collateral:

-   AA_PhaseInterpWithDelay.gds

-   AA_PhaseInterpWithDelay.cdl

-   AA_PhaseInterpWithDelay.sv

-   AA_PhaseInterpWithDelay.lef

-   AA_PhaseInterpWithDelay_shell.v

Lib file location:

gen_outputs/ip_blocks/dll_interpolator/dll_interpolator_tt_25_0p900_0p800.lib

**DLL Phase Det:**

Yaml file: dll_phasedet.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/dll_phasedet.yaml

Folder:

gen_outputs/dsn_phasedet_final

Generated collateral:

-   aib_phasedet_final.gds

-   aib_phasedet_final.cdl

-   aib_phasedet_final.sv

-   aib_phasedet_final.lef

-   aib_phasedet_final_shell.v

Lib file location:

gen_outputs/ip_blocks/dll_phasedet/dll_phasedet_tt_25_0p900_0p800.lib

**Frontend:**

Yaml file: frontend.yaml

Full command:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py
data/aib_ams/specs_dsn/frontend.yaml

Folder:

gen_outputs/aib_frontend

Generated collateral:

-   aib_frontend.gds

-   aib_frontend.cdl

-   aib_frontend.v

-   aib_frontend.lef

-   aib_frontend_shell.v

Lib file location:

gen_outputs/ip_blocks/frontend/frontend_tt_25_0p900_0p800.lib

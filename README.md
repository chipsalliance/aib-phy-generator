# AIB PHY Generator

Developed by Blue Cheetah Analog Design, Inc.
info@bcanalog.com


# Requirements
*   RHEL6 64-bit OS
*   git with internet connection
*   Cadence tools:
    *   Virtuoso
    *   Pegasus/PVS (for running LVS)
    *   Quantus/QRC (for running Extraction)
    *   SRR (for reading spectre data)
*   A full list of dependencies and versions is in Appendix A below.

*----------------------------------------
# Installing the AIB Generator

1. Clone the AIB Generator repository using a git command.  The following
commands use an example target directory of ~/aib. This must be done on a 
computer that has access to the internet.

cd ~/aib

Do a git init if you haven't already

git clone https://github.com/chipsalliance/aib-phy-generator aib-phy-generator

If your Linux machines cannot access the internet, you can use a Windows
computer that is connected to the internet.  Download "git for windows"
(gitforwindows.org) and use the gitbash shell in that shell.  You will need
a means to have the Windows machine mount your target directory used by the
Linux computer.

2. Initialize git submodules.  This must be done on a computer that has
access to the internet.

cd ~/aib/aib-phy-generator

If you need to switch submodules to use https instead of ssh, run the 
included ssh_to_https.sh script:

./ssh_to_https.sh

Otherwise run the recursive submodule update:

git submodule update --init --recursive

If you ran the git commands in Windows, you can now transfer the
aib-phy-generator folder to your linux machine to continue installation.

3. Download the Cadence Generic Process Design Kit into your target directory.
Use the web address shown in Table 1.  Cadence requires a login. Navigate to
the page "Generic Process Design Kits (GPDK) Downloads.”  Select and download
the file into your target ~/aib directory.

    ADVGPDK (Version 0.5) - Advanced Node 0.8V / 1.8V Finfet / Multi Patterned

    8 Metal Generic PDK: cds_ff_mpt_v_0.5 (1.2GB)**

List the file

cd ~/aib

ls cds_ff_mpt_v_0.5.tar.gz

cds_ff_mpt_v_0.5.tar.gz

4. From the target directory, extract from gzip and tar.

cd ~/aib

gunzip cds_ff_mpt_v_0.5.tar.gz

tar xvf cds_ff_mpt_v_0.5.tar

5. Acquire the packages in Appendix A and install.
Alternatively, you may contact Blue Cheetah Analog Design, Inc. for support at

    info@bcanalog.com

and request the "bag3d" tar file for aib-phy-generator. Install the Blue Cheetah file

bag3d0_rhel60_64_May29_2020.tar.gz

into your target directory, ~/aib in this example.

Unzip the bag3d file:

cd ~/aib

gunzip bag3d0_rhel60_64_may29_2020.tar.gz

Extract the bag3d0 files:

cd ~/aib/aib-phy-generator

tar xvf ../bag3d0_rhel60_64_may29_2020.tar

6. Edit `.bashrc` or `.cshrc` shell script so environment variables point to the
correct directory for various CAD tools.  Variables to change are:

CDS_INST_DIR: Virtuoso install directory.

PEGASUS_HOME: Pegasus/PVS install directory. If you have an alternate PVS license scheme, 
   comment this line out.

SPECTRE_HOME: Spectre install directory.

QRC_HOME: Quantus/QRC install directory.

OA_CDS_ROOT: OpenAccess directory packaged with Virtuoso.  Should be under $CDS_INST_DIR
             (the version number may be different depending on your Virtuoso install).

If you use a method other than setting "$CDS_LIC_FILE" for acquiring tool licenses,
comment out the last line.

7. update symbolic link to the PDK using the following commands:

cd ~/aib/aib-phy-generator/cds_ff_mpt/workspace_setup

rm PDK

ln -s ../../../cds_ff_mpt_v_0.5 PDK

8. Run the following setup script (source `.cshrc` instead in C-Shell):

cd ~/aib/aib-phy-generator

source .bashrc

./setup_script.sh


9. If you use a different command for PVS/Pegasus, change the bag_config.yaml file:

cd aib-phy-generator/cds_ff_mpt/workspace_setup


In the file bag_config.yaml, modify 

    lvs_cmd: 'pegasus'

To

    lvs_cmd: 'pvs'

10. If you performed the "git clone" commands on a Windows, machine, the symbolic link
in cds_ff_mpt will not be correct.  Perform the following commands from the 
aib-phy-generator directory:

cd cds_ff_mpt/pvs_setup

rm pvs_rules

ln -s ../workspace_setup/PDK/pvs/pvlLVS.rul pvs_rules

cd ../..


#
# Running the AIB Generator
1. The specs_dsn directory contains the design scripts that run through sizing procedures and 
simulations for creating the blocks.
- delay_line (dcc and dll)
- phase_interp (dcc and dll)
- phasedet (dcc and dll)
- dcc_helper
- frontend

There is a slight difference in the frontend because of its hierarchy on top of rxanlg and 
txanlg. There are separate rxanlg and txanlg design scripts mainly for debugging. The frontend 
design script expects that rxanlg has already been designed mainly due to the execution time 
of running through these scripts.

2. Perform any necessary commands in terminal so that you can acquire licenses for CAD tools, 
then source `.bashrc`/`.cshrc` in your terminal.

source .bashrc

3. Run the design script for designing the DLL Phase Interpolator:

./run_bag.sh BAG_framework/run_scripts/dsn_cell.py data/aib_ams/specs_dsn/dll_phase_interp.yaml

You may edit dll_phase_interp.yaml to produce a delay vs. step graph:


Change

  plot_result: False

To

  plot_result: True


You may replace "dll_phase_interp.yaml" with the following from data/aib_ams/specs_dsn:
- dcc_delay_line.yaml
- dcc_helper.yaml
- dcc_phase_interp.yaml
- dcc_phasedet.yaml
- dll_delay_line.yaml
- dll_phase_interp.yaml
- dll_phasedet.yaml
- frontend.yaml
- rxanlg.yaml
- txanlg.yaml

The outputs of the design phase are in directories under gen_outputs.

4. Run the IP generation script to produce gds, lef, cdl and Verilog models.

./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/aib_ams/specs_ip/dll_phase_interp.yaml -raw -mod -lef

You may replace "dll_phase_interp.yaml" with the following from data/aib_ams/specs_ip:
- dcc_delay_line.yaml
- dcc_helper.yaml
- dcc_phase_det.yaml
- dcc_phase_interp.yaml
- dll_delay_line.yaml
- dll_phase_interp.yaml
- dll_phasedet.yaml
- frontend.yaml
- rxanlg.yaml
- txanlg.yaml

The outputs of the IP generation are in directories under gen_outputs/ip_blocks.

5. Run the timing script to produce lib files.

./run_bag.sh bag3_digital/scripts_util/gen_lib.py data/aib_ams/specs_ip/dll_phase_interp_lib.yaml

You may replace "dll_phase_interp_lib.yaml" with the following from 
data/aib_ams/specs_ip:
- dcc_delay_line_lib.yaml
- dcc_helper_lib.yaml
- dcc_phase_det_lib.yaml
- dcc_phase_interp_lib.yaml
- dll_delay_line_lib.yaml
- dll_phase_interp_lib.yaml
- dll_phasedet_lib.yaml
- frontend_lib.yaml
- rxanlg_lib.yaml
- txanlg_lib.yaml

The output .lib files are in directories under gen_outputs/ip_blocks.


A full block by block overview can be seen [here](INSTRUCTIONS.md)


*----------------------------------------
# Appendix A: BAG Framework Dependencies & Versions

Python 3.7

Python packages:
- apipkg v.1.5
- appdirs v.1.4.3
- attrs v.19.3.0
- backcall v.0.1.0
- cycler v.0.10.0
- decorator v.4.4.2
- distlib v.0.3.0
- execnet v.1.7.1
- filelock v.3.0.12
- h5py v.2.10.0
- importlib-metadata v.1.6.0
- ipython v.7.13.0
- ipython-genutils v.0.2.0
- jedi v.0.16.0
- Jinja2 v.2.11.1
- kiwisolver v.1.2.0
- MarkupSafe v.1.1.1
- matplotlib v.3.2.1
- more-itertools v.8.2.0
- networkx v.2.4
- numpy v.1.18.2
- packaging v.20.3
- pandocfilters v.1.4.2
- parso v.0.6.2
- pexpect v.4.8.0
- pickleshare v.0.7.5
- pluggy v.0.13.1
- prompt-toolkit v.3.0.5
- ptyprocess v.0.6.0
- py v.1.8.1
- Pygments v.2.6.1
- pyparsing v.2.4.6
- PyQt5 v.5.14.0
- PyQt5-sip v.12.7.1
- pytest v.5.4.1
- pytest-forked v.1.1.3
- pytest-xdist v.1.31.0
- python-dateutil v.2.8.1
- pyzmq v.19.0.0
- ruamel.yaml v.0.16.10
- ruamel.yaml.clib v.0.2.0
- scipy v.1.4.1
- six v.1.14.0
- sortedcontainers v.2.1.0
- traitlets v.4.3.3
- virtualenv v.20.0.15
- wcwidth v.0.1.9
- zipp v.3.1.0
- zmq v.0.0.0

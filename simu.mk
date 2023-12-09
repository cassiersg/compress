
WORKDIR ?= ./
IVERILOG_WARNINGS ?=

export SYNTH_SRCS
export VINCLUDE
export VERILOG_SOURCES
export TOPLEVEL
export STATS
export NUM_SHARES
export CIRCUIT_FILE_PATH
#

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_INCLUDE_DIRS=$(VINCLUDE)
COMPILE_ARGS += $(IVERILOG_WARNINGS)
SIM_BUILD=$(WORKDIR)/work
export COCOTB_RESULTS_FILE = $(WORKDIR)/result.xml

# MODULE is the basename of the Python test file
MODULE = scripts.sim_tb

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim


WORKDIR ?= ./
IVERILOG_WARNINGS ?=

PLUSARGS += -fst 
VERILOG_SOURCES += $(WORKDIR)/iverilog_dump.v
COMPILE_ARGS += -s iverilog_dump

$(WORKDIR)/iverilog_dump.v:
	echo 'module iverilog_dump();' > $@
	echo 'initial begin' >> $@
	echo '    $$dumpfile("$(WORKDIR)/sim-dump.fst");' >> $@
	echo '    $$dumpvars(0, $(TOPLEVEL));' >> $@
	echo 'end' >> $@
	echo 'endmodule' >> $@

COCOTB_REDUCED_LOG_FMT=0

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

# Used to avoid recursive make call from cocotb Makefile when running 'sim' target
.PHONY: sim_nonrecursive
sim_nonrecursive: $(COCOTB_RESULTS_FILE)


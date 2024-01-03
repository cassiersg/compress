
WORK ?= ./work
CIRCUIT ?= circuits/ascon.txt
GADGETS_CONFIG ?= ./gadget_library/gadgets_opt.toml
LATS ?=4 5 6
DS ?= 2 3 4 5

export SYNTH_LIB=synthesis/stdcells.lib
export VERILOG_CELLS=synthesis/stdcells.v
export SYNTH_SRCS=gadget_library/BIN/*.v gadget_library/MSK/*.v gadget_library/RNG/trivium/*.v
export VINCLUDE=gadget_library/MSK

CIRCUIT_NAME=$(patsubst %.txt,%,$(notdir $(CIRCUIT)))

PYTHON ?= python3
SCRIPT_DIR=scripts
COMPRESS_SCRIPT=$(SCRIPT_DIR)/compress.py
SYNTH_SCRIPT=synthesis/synth.tcl

GADGETS=$(basename $(notdir $(wildcard gadget_library/MSK/*.v)))


GADGET_AREA_FILE = $(WORK)/gadget_area/$(1)_d$(2)/area.json
GADGETS_AREA_CSV = $(WORK)/gadget_area/areas_d$(1).csv

define rules_gadget_area

$(call GADGET_AREA_FILE,%,$(1)): gadget_library/MSK/%.v $(SYNTH_SCRIPT)
	@mkdir -p $$(dir $$@)
	VDEFINES="-DDEFAULTSHARES=$(1)" \
	SYNTH_TOP=$$* \
	RESDIR=$$(dir $$@) \
	yosys -qq -c $(SYNTH_SCRIPT) -l  $$(dir $$@)/synth.log

$(call GADGETS_AREA_CSV,$(1)): $(foreach G,$(GADGETS),$(call GADGET_AREA_FILE,$G,$(1)))
	$(PYTHON) $(SCRIPT_DIR)/summarize_gadget_area.py --outcsv=$$@ $$^

endef
$(foreach D,$(DS),$(eval $(call rules_gadget_area,$D)))

$(WORK)/rng_area/nbits_%/area.json: gadget_library/RNG/trivium/*.v
	@mkdir -p $(dir $@)
	VDEFINES="-DDEFAULTRND=$*" \
	SYNTH_TOP=prng_top \
	RESDIR=$(dir $@) \
	yosys -qq -c $(SYNTH_SCRIPT) -l  $(dir $@)/synth.log

RNG_AREA = $(WORK)/rng_area/area_ge.txt
$(RNG_AREA): $(foreach N,32 512,$(WORK)/rng_area/nbits_$N/area.json)
	$(PYTHON) $(SCRIPT_DIR)/rng_area.py --out=$@ $^


circuit_dir = $(WORK)/circuits/$(CIRCUIT_NAME)_d$(1)_l$(2)
circuit_nshares = $(shell echo $(1) | sed -n "s/.*_d\([0-9]*\)_l\([0-9]*\)$$/\1/p")
circuit_latency = $(shell echo $(1) | sed -n "s/.*_d\([0-9]*\)_l\([0-9]*\)$$/\2/p")

CIRCUIT_DIRS = $(foreach D,$(DS),$(foreach L,$(LATS),$(call circuit_dir,$D,$L)))

# COMPRESS
define rule_compress
$(call circuit_dir,$(1),$(2))/design.v: $(CIRCUIT) $(COMPRESS_SCRIPT) $(RNG_AREA) $(call GADGETS_AREA_CSV,$(1))
	@mkdir -p $$(dir $$@)
	$(PYTHON) $(COMPRESS_SCRIPT) \
		--num-shares $(1) --latency=$(2) \
		--circuit=$$< \
		--out=$$@ --outh=$$@h \
		--gadgets-area-csv=$(WORK)/gadget_area/areas_d$(1).csv --rng-area-txt=$(RNG_AREA) \
		--outstats=$$(dir $$@)/stats.json \
		--gadgets-config=$(GADGETS_CONFIG) \
		--time-limit 3600 \
		> $$(dir $$@)/compress.log
	cp $$@ $$(dir $$@)/$(CIRCUIT_NAME).v
	-cp $$@h $$(dir $$@)/$(CIRCUIT_NAME).vh
endef
$(foreach D,$(DS),$(foreach L,$(LATS),$(eval $(call rule_compress,$D,$L))))

# SIMULATIONS

# behavioral simu
$(addsuffix /beh_simu/simu.log,$(CIRCUIT_DIRS)) : %/beh_simu/simu.log : %/design.v
	@mkdir -p $(dir $@)
	IVERILOG_WARNINGS="-Wimplicit -Wportbind -Wselect-range -Winfloop" \
					  VERILOG_SOURCES="$(SYNTH_SRCS) $<" \
					  STATS=$(dir $@)/../stats.json \
					  TOPLEVEL=$(CIRCUIT_NAME) \
					  CIRCUIT_FILE_PATH=$(CIRCUIT) \
					  WORKDIR=$(dir $@) \
					  $(MAKE) -f simu.mk sim_nonrecursive > $@ || exit 0

# structural simu
$(addsuffix /struct_simu/simu.log,$(CIRCUIT_DIRS)): %/struct_simu/simu.log: %/synth/design.v
	@mkdir -p $(dir $@)
	VERILOG_SOURCES="$< $(VERILOG_CELLS)" \
					STATS=$(dir $@)/../stats.json \
					  TOPLEVEL=$(CIRCUIT_NAME) \
					  CIRCUIT_FILE_PATH=$(CIRCUIT) \
					  WORKDIR=$(dir $@) \
					  $(MAKE) -f simu.mk sim_nonrecursive > $@ || exit 0


# Mark simulation success (simulation always return a zero exit code).
%/success: %/simu.log
	grep -q -s FAIL=0 $< && touch $@ || exit 1

# SYNTHESIS
ALL_DESIGNS = $(addsuffix /synth/design.v,$(CIRCUIT_DIRS))
ALL_AREAS = $(addsuffix /synth/area.json,$(CIRCUIT_DIRS))
$(ALL_DESIGNS): %/synth/design.v: %/design.v $(SYNTH_SCRIPT) | %/beh_simu/success
	@mkdir -p $(dir $@)
	SYNTH_SRCS="$(SYNTH_SRCS) $<" SYNTH_TOP=$(CIRCUIT_NAME) VDEFINES="-DDEFAULTSHARES=$(call circuit_nshares,$*)" \
	RESDIR=$(dir $@) \
	yosys -qq -c $(SYNTH_SCRIPT) -l  $(dir $@)/synth.log

$(ALL_AREAS): %/area.json: %/design.v

ALL_FLOWS=$(addsuffix /struct_simu/success,$(CIRCUIT_DIRS))

AREA_REPORT = $(WORK)/$(CIRCUIT_NAME)_area.csv
$(AREA_REPORT): $(ALL_FLOWS)
	$(PYTHON) $(SCRIPT_DIR)/summarize_design_area.py --outcsv=$@ $(ALL_AREAS)

area: $(AREA_REPORT)

clean:
	-rm -r $(WORK)/

.PHONY: clean area

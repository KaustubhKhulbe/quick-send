SHELL=/bin/bash -o pipefail
.SHELLFLAGS += -e

export PKG_SRCS = $(PWD)/../pkg/types.sv
export HDL_SRCS = $(shell find $(PWD)/../hdl -name '*.sv')
export SRAM_LIB = $(shell find $(PWD)/../sram/output/ -name "*.db")
export DESIGN_TOP = bp

export STD_CELL_LIB=$(FREEPDK45)/stdcells.db
export STD_CELL_ALIB=$(FREEPDK45)/alib

export SNPSLMD_QUEUE=true

.PHONY: synth
synth: clean
	$(MAKE) outputs/synth.ddc

outputs/synth.ddc:
	mkdir -p reports outputs
	python3 $(PWD)/../bin/get_options.py clock
	python3 $(PWD)/../bin/get_options.py min_power
	python3 $(PWD)/../bin/get_options.py synth_cmd
	python3 $(PWD)/../bin/get_options.py synth_cmd_inc
	python3 $(PWD)/../bin/get_options.py synth_inc_iter
	export ECE411_CLOCK_PERIOD_PS=$(shell python3 $(PWD)/../bin/get_options.py clock) ;\
	export ECE411_MIN_POWER=$(shell python3 $(PWD)/../bin/get_options.py min_power) ;\
	export ECE411_COMPILE_CMD="$(shell python3 $(PWD)/../bin/get_options.py synth_cmd)" ;\
	export ECE411_COMPILE_CMD_INC="$(shell python3 $(PWD)/../bin/get_options.py synth_cmd_inc)" ;\
	export ECE411_COMPILE_ITER="$(shell python3 $(PWD)/../bin/get_options.py synth_inc_iter)" ;\
	export ECE411_DC_CORES=4 ;\
	dc_shell -f synthesis.tcl |& tee reports/synthesis.log
	rm -f  *.log
	rm -f  default.svf
	rm -rf work
	bash check_synth_error.sh

.PHONY: dv
dv:
	timeout $(ECE411_GUI_TIMEOUT) design_vision -f dv.tcl

.PHONY: clean
clean:
	rm -f  *.log
	rm -f  default.svf
	rm -rf work
	rm -rf reports
	rm -rf outputs

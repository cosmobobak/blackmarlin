EXE    	= BlackMarlin
EVALFILE = ./nn/default.bin
ifeq ($(OS),Windows_NT)
NAME := $(EXE).exe
else
NAME := $(EXE)
endif

rule:
	EVALFILE=$(EVALFILE) cargo rustc --release --features nnue -- -C target-cpu=native --emit link=$(NAME)
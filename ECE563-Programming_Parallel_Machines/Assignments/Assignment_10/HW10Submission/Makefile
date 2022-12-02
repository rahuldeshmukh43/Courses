SHELL := /bin/bash

#CONDA_HOME := $(shell dirname $(shell dirname $(shell which conda)))
#ENV := $(shell basename $(shell dirname $(shell dirname $(shell which python))))
#BIN_DIR :=$(CONDA_HOME)/envs/$(ENV)/bin/
#LIB_DIR :=$(CONDA_HOME)/envs/$(ENV)/lib/
#INCLUDE_DIR :=$(CONDA_HOME)/envs/$(ENV)/include/

#NVCC :=  $(BIN_DIR)nvcc
NVCC :=  nvcc
CFLAGS := -std=c++17 -Wall -pedantic -O3
OMPFLAG := -fopenmp
DEBUG_FLAG := -g

all: hello_world dotproduct device_prop

clean:
	rm hello_world
	rm dotproduct
	rm device_properties

#echo:
#	@echo CONDA HOME : $(CONDA_HOME)
#	@echo ENV : $(ENV)
#	@echo BIN DIR: $(BIN_DIR) 
#	@echo LIB DIR : $(LIB_DIR)
#	@echo INCLUDE DIR : $(INCLUDE_DIR)
#	@echo NVCC : $(NVCC)
#	@echo --------------------------------------------------
#	@echo

hello_world:
	$(NVCC) -x cu  hello_world.cpp -o hello_world
	
device_prop:
	$(NVCC) -x cu device_properties.cpp -o device_properties

dotproduct:
	$(NVCC) -x cu -Xcompiler $(OMPFLAG) dotproduct.cpp -o dotproduct

ifneq ($(KERNELRELEASE),)
	obj-m += hello.o
	ccflags-y := -std=gnu11 -Wno-declaration-after-statement
else
	KERNELDIR ?= /lib/modules/$(shell uname -r)/build
	PWD := $(shell pwd)
default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
	find . -maxdepth 1 -type f | grep -P "\.(cmd|symvers|order|mod.c|o|mod)$$" | xargs rm -f {}
endif

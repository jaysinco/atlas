ifneq ($(KERNELRELEASE),)
	obj-m += hello.o
	obj-m += test.o
else
	KERNELDIR ?= /lib/modules/$(shell uname -r)/build
	PWD := $(shell pwd)
default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
clean:
	find . -maxdepth 1 -type f | grep -P "\.(cmd|symvers|order|mod.c|o|ko|mod)$$" | xargs rm -f {}
endif

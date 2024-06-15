#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("sinco");
MODULE_DESCRIPTION("A simple Hello world!");
MODULE_VERSION("0.1");

static int __init hello_init(void)
{
    printk(KERN_INFO "Hello, world\n");
    printk(KERN_INFO "The process is \"%s\" (pid %i)\n", current->comm, current->pid);
    return 0;
}

static void __exit hello_exit(void) { printk(KERN_INFO "Goodbye, world\n"); }

module_init(hello_init);
module_exit(hello_exit);

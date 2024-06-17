#include <linux/init.h>
#include <linux/fs.h>
#include <linux/module.h>
#include <linux/cdev.h>

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("sinco");
MODULE_DESCRIPTION("A simple Hello world!");
MODULE_VERSION("0.1");

struct my_dev
{
    struct mutex lock;
    struct cdev cdev;
};

int nr_devs = 1;
int my_major;
int my_minor = 0;
struct my_dev* my_devices;

loff_t mySeek(struct file* filp, loff_t off, int whence)
{
    struct scull_dev* dev = filp->private_data;
    loff_t newpos;

    switch (whence) {
        case 0: /* SEEK_SET */
            newpos = off;
            break;
        case 1: /* SEEK_CUR */
            newpos = filp->f_pos + off;
            break;
        case 2: /* SEEK_END */
            newpos = off;
            break;
        default: /* can't happen */
            return -EINVAL;
    }
    if (newpos < 0) {
        return -EINVAL;
    }
    filp->f_pos = newpos;
    return newpos;
}

ssize_t myRead(struct file* filp, char __user* buf, size_t count, loff_t* f_pos)
{
    printk(KERN_INFO "read device %zu\n", count);
    return -EPERM;
}

ssize_t myWrite(struct file* filp, char __user const* buf, size_t count, loff_t* f_pos)
{
    printk(KERN_INFO "write device %zu\n", count);
    return -EPERM;
}

int myOpen(struct inode* inode, struct file* filp)
{
    struct my_dev* dev;
    dev = container_of(inode->i_cdev, struct my_dev, cdev);
    filp->private_data = dev;
    return 0;
}

int myRelease(struct inode* inode, struct file* filp) { return 0; }

struct file_operations my_fops = {
    .owner = THIS_MODULE,
    .llseek = mySeek,
    .read = myRead,
    .write = myWrite,
    .open = myOpen,
    .release = myRelease,
};

static void addDev(struct my_dev* dev, int major, int minor, int index)
{
    int err, devno = MKDEV(major, minor + index);

    cdev_init(&dev->cdev, &my_fops);
    dev->cdev.owner = THIS_MODULE;
    err = cdev_add(&dev->cdev, devno, 1);
    if (err) {
        printk(KERN_ERR "failed to add device %d:%d\n", index, err);
        return;
    }
    printk(KERN_INFO "add device %d:%d\n", major, minor + index);
}

static int __init myInit(void)
{
    int i;
    int err;
    dev_t dev;

    printk(KERN_INFO "Hello, world\n");
    printk(KERN_INFO "The process is \"%s\" (pid %i)\n", current->comm, current->pid);

    err = alloc_chrdev_region(&dev, my_minor, nr_devs, "hello");
    my_major = MAJOR(dev);
    if (err < 0) {
        printk(KERN_ERR "faild to get major %d\n", my_major);
        return err;
    }

    my_devices = kmalloc(nr_devs * sizeof(struct my_dev), GFP_KERNEL);
    if (!my_devices) {
        err = -ENOMEM;
        return err;
    }
    memset(my_devices, 0, nr_devs * sizeof(struct my_dev));

    for (i = 0; i < nr_devs; i++) {
        addDev(&my_devices[i], my_major, my_minor, i);
    }

    return 0;
}

static void __exit myExit(void)
{
    int i;
    dev_t devno;

    if (my_devices) {
        for (i = 0; i < nr_devs; i++) {
            cdev_del(&my_devices[i].cdev);
        }
        kfree(my_devices);
    }

    devno = MKDEV(my_major, my_minor);
    unregister_chrdev_region(devno, nr_devs);

    printk(KERN_INFO "Goodbye, world\n");
}

module_init(myInit);
module_exit(myExit);

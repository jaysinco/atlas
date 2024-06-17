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
struct class* my_class;
struct my_dev* my_devices;

loff_t mySeek(struct file* filp, loff_t off, int whence)
{
    // struct scull_dev* dev = filp->private_data;
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
        default:
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
    printk(KERN_INFO "hello: read device %zu\n", count);
    return -EPERM;
}

ssize_t myWrite(struct file* filp, char __user const* buf, size_t count, loff_t* f_pos)
{
    printk(KERN_INFO "hello: write device %zu\n", count);
    return -EPERM;
}

int myOpen(struct inode* inode, struct file* filp)
{
    struct my_dev* dev = container_of(inode->i_cdev, struct my_dev, cdev);
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

static void addDev(struct my_dev* dev, struct class* cls, int major, int minor, int index)
{
    int devno = MKDEV(major, minor + index);
    cdev_init(&dev->cdev, &my_fops);
    dev->cdev.owner = THIS_MODULE;
    int err = cdev_add(&dev->cdev, devno, 1);
    if (err) {
        printk(KERN_ERR "hello: failed to add device %d:%d\n", index, err);
        return;
    }
    if (cls) {
        struct device* d = device_create(cls, NULL, devno, NULL, "hello%d", index);
        if (d == NULL) {
            printk(KERN_WARNING "failed to create device\n");
        }
    }
    printk(KERN_INFO "hello: add device %d:%d\n", major, minor + index);
}

static int __init myInit(void)
{
    printk(KERN_INFO "hello: Hello, world\n");
    printk(KERN_INFO "hello: The process is \"%s\" (pid %i)\n", current->comm, current->pid);

    dev_t dev;
    int err = alloc_chrdev_region(&dev, my_minor, nr_devs, "hello");
    my_major = MAJOR(dev);
    if (err < 0) {
        printk(KERN_ERR "hello: faild to get major %d\n", my_major);
        return err;
    }

    my_devices = kmalloc(nr_devs * sizeof(struct my_dev), GFP_KERNEL);
    if (!my_devices) {
        return -ENOMEM;
    }
    memset(my_devices, 0, nr_devs * sizeof(struct my_dev));

    my_class = class_create(THIS_MODULE, "hello_dev");
    if (my_class == NULL) {
        printk(KERN_WARNING "failed to create class\n");
    }

    for (int i = 0; i < nr_devs; i++) {
        addDev(&my_devices[i], my_class, my_major, my_minor, i);
    }

    return 0;
}

static void __exit myExit(void)
{
    if (my_devices) {
        for (int i = 0; i < nr_devs; i++) {
            if (my_class) {
                device_destroy(my_class, my_devices[i].cdev.dev);
            }
            cdev_del(&my_devices[i].cdev);
        }
        kfree(my_devices);
    }
    if (my_class) {
        class_destroy(my_class);
    }

    dev_t devno = MKDEV(my_major, my_minor);
    unregister_chrdev_region(devno, nr_devs);

    printk(KERN_INFO "hello: Goodbye, world\n");
}

module_init(myInit);
module_exit(myExit);

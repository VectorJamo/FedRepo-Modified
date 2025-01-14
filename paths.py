import os

data_dir = r"C:\Workspace\work\datasets"
cur_dir = "./"

# Modify this to the directory where you have your data
if not os.path.exists(data_dir):
    data_dir = os.path.join(os.getcwd(), "raw-data") # *CODE MODIFIED HERE*

# Make sure to create sub-folders with these names and put the respective data there
cifar_fdir = os.path.join(data_dir, "Cifar")
famnist_fdir = os.path.join(data_dir, "Fasion-MNIST")

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cifar_fpaths = {
    "cifar10": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar10-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar10-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar10-test.pkl")
    },
    "cifar100": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar100-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar100-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar100-test.pkl")
    },
}

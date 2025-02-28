import os

data_dir = r"C:\Workspace\work\datasets"
cur_dir = "./"

# Modify this to the directory where you have your data
if not os.path.exists(data_dir):
    data_dir = os.path.join(os.getcwd(), "raw-data") # *CODE MODIFIED HERE*

# Make sure to create sub-folders with these names and put the respective data there
cifar_fdir = os.path.join(data_dir, "Cifar")
mnist_fdir = os.path.join(data_dir, "Mnist")
famnist_fdir = os.path.join(data_dir, "Mnist")


tumor_fdir = os.path.join(data_dir, "Tumor")
behavior_fdir = os.path.join(data_dir, "Behavioral")

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
mnist_fpaths = {
    "mnist": {
        "train_fpaths": [
            os.path.join(mnist_fdir, "mnist_train.pkl"),
        ],
        "test_fpath": os.path.join(mnist_fdir, "mnist_test.pkl")
    },
    "fmnist": {
        "train_fpaths": [
            os.path.join(mnist_fdir, "famnist_train.pkl"),
        ],
        "test_fpath": os.path.join(mnist_fdir, "famnist_test.pkl")
    },
}

cifar_fpaths = {
    "cifar10": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar10_train_modified.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar10_test_modified.pkl")
    },
    "cifar100": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar100-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar100-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar100-test.pkl")
    },
}

tumor_fpaths = {
    "tumor4": {
        "train_fpaths": [
            os.path.join(tumor_fdir, "tumor4train.pkl"),
        ],
        "test_fpath":[
            os.path.join(tumor_fdir, "tumor4test.pkl")
            ]
    },
    "tumor2": {
        "train_fpaths": [
            os.path.join(tumor_fdir, "tumor2train.pkl"),
        ],
        "test_fpath":[
            os.path.join(tumor_fdir, "tumor2test.pkl")
            ]
    },
}

behavioral_fpaths = {
    "DAC": {
        "train_fpaths": [
            os.path.join(behavior_fdir, "DACtrain_data.pkl"),
        ],
        "test_fpath":[
            os.path.join(behavior_fdir, "DACtest_data.pkl")
            ]
    },
    "Swipe": {
        "train_fpaths": [
            os.path.join(behavior_fdir, "Swipetrain_data.pkl"),
        ],
        "test_fpath":[
            os.path.join(behavior_fdir, "Swipetest_data.pkl")
            ]
    },
    "Voice": {
        "train_fpaths": [
            os.path.join(behavior_fdir, "Voicetrain_data.pkl"),
        ],
        "test_fpath":[
            os.path.join(behavior_fdir, "Voicetest_data.pkl")
            ]
    },
}

import kagglehub

# Download ImageNet
#path = kagglehub.dataset_download("sautkin/imagenet1k0") #train 0-499 classes
#path = kagglehub.dataset_download("sautkin/imagenet1k1") #train 500-999 classes
#path = kagglehub.dataset_download("sautkin/imagenet1k2") #test 0-499 classes
path = kagglehub.dataset_download("sautkin/imagenet1k3") #test 500-999 classes
#path = kagglehub.dataset_download("sautkin/imagenet1kvalid") #val 0-999 classes

print("Path to dataset files:", path)
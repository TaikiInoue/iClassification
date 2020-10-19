import urllib.request
from urllib.request import urlretrieve

url_dict = {
    "resnet50": "https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet50_batch256_20200708-cfb998bf.pth",
    "resnet101": "https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet101_batch256_20200708-753f3608.pth",
}

# for model_name, url in url_dict.items():
#     urlretrieve(url, f"{model_name}_original.pth")


response = urllib.request.urlopen(
    "https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet101_batch256_20200708-753f3608.pth"
)
state_dict = response.read()
print(state_dict.keys())

from cifar_lib_0 import *

data_dir = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

input_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
target_path = os.path.join(data_dir, "cifar-10-python.tar")


if not os.path.exists(input_path):
    urllib.request.urlretrieve(url, input_path)

    with gzip.open(input_path, 'rb') as f_in:
        with open(target_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    tar = tarfile.TarFile(target_path)
    tar.extractall(data_dir)
    tar.close
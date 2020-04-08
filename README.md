## CNNDroid

### 环境配置
#### apktool配置

apktool需要java环境，java 1.8.0_191版本会报错，测试发现java version "11.0.1" 2018-10-16 LTS以及上版本可用

配置java环境

```shell
tar -xvf jdk-11.0.1_linux-x64_bin.tar.gz
chmod +x -R jdk-11.0.1
mkdir java_env
mv jdk-11.0.1 java_env/
```

添加或修改配置文件.bashrc，path_to_java_env为文件夹java_env的路径
```
#java_env
export JAVA_HOME=path_to_java_env/jdk-11.0.1
export JRE_HOME=$JAVA_HOME/jre
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
export CLASSPATH=$CLASSPATH:$JAVA_HOME/lib:$JRE_HOME/lib:.
```

加载配置文件

```bash
source ~/.bashrc
```

查看当前java版本

```bash
java -version
```



#### anaconda配置

安装anaconda3 python3.7

```shell
bash Anaconda3-2019.10-Linux-x86_64.sh
```

更换为清华的源

```bash
#运行以下命令
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
#或者修改用户目录下的 .condarc文件为以下内容
ssl_verify: true
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/simpleitk

show_channel_urls: true
```

#### pytorch配置

查看CUDA版本

```bash
cat /usr/local/cuda/version.txt
```

安装对应与CUDA版本的pytorch 1.0.0

```bash
# CUDA 10.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch

# CUDA 9.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch

# CUDA 8.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch

# CPU Only
conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch
```

查看pytorch版本和GPU是否可用

```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

#### 其他依赖

torchnet

```
pip install torchnet
```

python-igraph

```
pip install python-igraph
```
fire
```
conda install fire
```
visdom
```
conda install visdom
```
pynvml
```
conda install pynvml
```
额外

```
pip install python-louvain
pip install tensorwatch
```

### 使用手册


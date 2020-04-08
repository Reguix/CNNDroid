# coding:utf8
import warnings
import torch as t
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/home/zhangxin/CNNDroid/Dataset/2014/image_800'  # 训练集存放路径
    #train_data_root = 'F:\\test\\decompileDataset\\image'  # 训练集存放路径
    test_data_root = '/home/zhangxin/CNNDroid/Dataset/2018/image_800/'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 16  # batch size
    use_gpu = True if t.cuda.is_available() else False # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 1000
    lr = 0.0004  # initial learning rate
    lr_decay = 0.85  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    
    val_accuracy_max = 0.94


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribut %s' % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()

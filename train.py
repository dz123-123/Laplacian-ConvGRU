from option import args, parser
import torch.backends.cudnn as cudnn

#################### Distributed learning setting #######################
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
#########################################################################

import torch.utils.data

from datasets_list import MyDataset

from utils import *

from trainer import train_net
from model import LDRN

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    #设置当前 GPU 编号
    args.multigpu = False
    # 初始化多 GPU 标志为 False
    if args.distributed:
    # 如果是分布式训练
        args.multigpu = True
    # 设置多 GPU 标志为 True
        args.rank = args.rank * ngpus_per_node + gpu
    # 计算当前进程的全局排名
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                 world_size=args.world_size, rank=args.rank)
    # 初始化分布式进程组
        args.batch_size = int(args.batch_size/ngpus_per_node)
    # 调整每个 GPU 的 batch_size
        args.workers = int((args.num_workers + ngpus_per_node - 1)/ngpus_per_node)
    # 调整每个 GPU 的 worker 数量
        print("==> gpu:",args.gpu,", rank:",args.rank,", batch_size:",args.batch_size,", workers:",args.workers)
        torch.cuda.set_device(args.gpu)
    # 设置当前 GPU
    elif args.gpu is None:
    # 如果没有指定 GPU
        print("==> DataParallel Training")
        args.multigpu = True
    # 设置多 GPU 标志为 True
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num
    # 设置可见的 GPU 设备编号
    else:
    # 单 GPU 训练
        print("==> Single GPU Training")
    # 设置当前 GPU
        torch.cuda.set_device(args.gpu)
    #断言是否启用cudnn后端
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    #根据参数和解析器格式化保存路径
    save_path = save_path_formatter(args, parser)
    # 设置保存模型的路径
    args.save_path = 'checkpoints'/save_path
    if (args.rank == 0):
    # 如果是主进程
        print('=> number of GPU: ',args.gpu_num)
    # 打印 GPU 数量
        print("=> information will be saved in {}".format(args.save_path))
    # 打印信息保存路径
    args.save_path.makedirs_p()
    # 创建保存路径
    torch.manual_seed(args.seed)
    # 设置随机种子
    ##############################    Data loading part    ################################
    if args.dataset == 'KITTI':
    # 根据数据集选择最大深度
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0
    #数据集实例化
    train_set = MyDataset(args, train=True)
    test_set = MyDataset(args, train=False)
    #打印数据集名称、数据的高度和宽度、训练集和测试集的样本数量。
    if (args.rank == 0):
        print("=> Dataset: ",args.dataset)
        print("=> Data height: {}, width: {} ".format(args.height, args.width))
        print('=> train samples_num: {}  '.format(len(train_set)))
        print('=> test  samples_num: {}  '.format(len(test_set)))
    #初始化采样器
    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    #这里val_loader实际上是测试集的数据加载器，但通常称为验证集加载器
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
    cudnn.benchmark = True
    #########################################################################################

    ###################### Setting Network, Loss, Optimizer part ###################
    #当前是主进程
    if (args.rank == 0):
        print("=> creating model")
    Model = LDRN(args)
    ############################### Number of model parameters ##############################
    #初始化两个变量来分别计数编码器和解码器的参数数量。
    num_params_encoder = 0
    num_params_decoder = 0
    #遍历编码器的所有参数，并计算它们的总数（numel()方法返回参数中元素的数量）。类似地，遍历解码器的所有参数，并计算它们的总数。
    for p in Model.encoder.parameters():
        num_params_encoder += p.numel()
    for p in Model.decoder.parameters():
        num_params_decoder += p.numel()
    #如果当前是主进程，则打印一条分隔线，然后分别打印出编码器和解码器的参数数量，以及它们的总和。
    if (args.rank == 0):
        print("===============================================")
        print("model encoder parameters: ", num_params_encoder)
        print("model decoder parameters: ", num_params_decoder)
        print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
        #遍历模型的所有参数，只考虑那些需要梯度（即可训练的）参数。对于每个可训练参数，使用np.prod(p.shape)计算其元素总数（即所有维度的大小乘积）。将所有可训练参数的元素总数相加，得到可训练参数的总数
        trainable_params = sum([np.prod(p.shape) for p in Model.parameters() if p.requires_grad])
        #打印出可训练参数的总数。
        print("Total trainable parameters: {}".format(trainable_params))
        print("===============================================")
    ############################### apex distributed package wrapping ########################
    #这段代码主要处理了在不同的训练设置（分布式训练、单GPU训练、以及多GPU但非分布式训练）下如何初始化和配置模型。
    if args.distributed:
        #BN归一化操作
        if args.norm == 'BN':
            Model = nn.SyncBatchNorm.convert_sync_batchnorm(Model)
            if (args.rank == 0):
                print("=> use SyncBatchNorm")
        Model = Model.cuda(args.gpu)
        Model = torch.nn.parallel.DistributedDataParallel(Model, device_ids=[args.gpu], output_device=args.gpu,
                                                           find_unused_parameters=True)
        print("=> Model Initialized on GPU: {} - Distributed Traning".format(args.gpu))
        enc_param = Model.module.encoder.parameters()
        dec_param = Model.module.decoder.parameters()
    elif args.gpu is None:
        Model = Model.cuda()
        Model = torch.nn.DataParallel(Model)
        print("=> Model Initialized - DataParallel")
        enc_param = Model.module.encoder.parameters()
        dec_param = Model.module.decoder.parameters()
    else:
        Model = Model.cuda(args.gpu)
        print("=> Model Initialized on GPU: {} - Single GPU training".format(args.gpu))
        enc_param = Model.encoder.parameters()
        dec_param = Model.decoder.parameters()
    
    ###########################################################################################

    ################################ pretrained model loading #################################
    #它涉及到模型的加载、优化器的设置、以及训练过程的启动。
    if args.model_dir != '':
        #Model.load_state_dict(torch.load(args.model_dir,map_location='cuda:'+args.gpu_num))
        Model.load_state_dict(torch.load(args.model_dir))
        if (args.rank == 0):
            print('=> pretrained model is created')
    #############################################################################################


    ############################## optimizer and criterion setting ##############################
    optimizer = torch.optim.AdamW([{'params': Model.module.encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   {'params': Model.module.decoder.parameters(), 'weight_decay': 0, 'lr': args.lr}], eps=args.adam_eps)
    ##############################################################################################
    logger = None

    ####################################### Training part ##########################################

    if (args.rank == 0):
        print("training start!")

    loss = train_net(args, Model, optimizer, train_loader,val_loader, args.epochs,logger)

    if (args.rank == 0):
        print("training is finished")

if __name__ == '__main__':
    args.batch_size_dist = args.batch_size
    args.num_threads = args.workers
    args.world_size = 1
    args.rank = 0
    nodes = "127.0.0.1"
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        print("==> Distributed Training")
        mp.set_start_method('forkserver')

        print("==> Initial rank: ",args.rank)
        port = np.random.randint(10000, 10030)
        args.dist_url = 'tcp://{}:{}'.format(nodes, port)
        print("==> dist_url: ",args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None
        args.workers = 9
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs = ngpus_per_node, args = (ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)

'''
# 检查是否指定了模型目录，如果有，则加载预训练模型  
if args.model_dir != '':  
    # 加载模型状态字典，这里注释掉的代码显示了如何指定加载到特定的CUDA设备上  
    # Model.load_state_dict(torch.load(args.model_dir, map_location='cuda:'+args.gpu_num))  
    # 直接加载模型状态字典，不指定设备，PyTorch会自动处理设备问题  
    Model.load_state_dict(torch.load(args.model_dir))  
    # 如果当前是主进程（rank为0），则打印消息  
    if (args.rank == 0):  
        print('=> pretrained model is created')  
  
# ############################################################  
# 优化器和损失函数设置  
# 使用AdamW优化器，对编码器和解码器设置不同的weight_decay  
optimizer = torch.optim.AdamW([  
    {'params': Model.module.encoder.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},  
    {'params': Model.module.decoder.parameters(), 'weight_decay': 0, 'lr': args.lr}  
], eps=args.adam_eps)  
  
# 假设logger变量用于记录训练过程，但这里初始化为None，可能后续会配置  
logger = None  
  
# ##########################################################  
# 训练部分  
# 如果当前是主进程（rank为0），则打印训练开始的消息  
if (args.rank == 0):  
    print("training start!")  
  
# 调用train_net函数进行训练，传入相关参数  
# 注意：train_net函数的具体实现不在此段代码中，但我们可以推测它负责训练过程  
# train_loader和val_loader分别是训练和验证的数据加载器  
# args.epochs指定了训练的轮数  
loss = train_net(args, Model, optimizer, train_loader, val_loader, args.epochs, logger)  
  
# 如果当前是主进程（rank为0），则打印训练完成的消息  
if (args.rank == 0):  
    print("training is finished")
'''




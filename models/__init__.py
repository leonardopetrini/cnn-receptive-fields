from .alexnet import *
from .vgg import *
from .resnet import *

def select_net(args):
    num_ch = 1 if 'mnist' in args.dataset else 3
    nc = 200 if 'tiny' in args.dataset else 10
    nc = 2 if 'diffeo' in args.dataset else nc
    num_classes = 1 if args.loss == 'hinge' else nc
    imsize = 28 if 'mnist' in args.dataset else 32
    try:
        args.fcwidth
    except:
        args.fcwidth = 64
    try:
        args.width
    except:
        args.width = args.fcwidth
    try:
        args.pretrained
    except:
        args.pretrained = 0
    if not args.pretrained:
        if 'VGG' in args.net:
            if 'bn' in args.net:
                bn = True
                net_name = args.net[:-2]
            else:
                net_name = args.net
                bn = False
            net = VGG(net_name, num_ch=num_ch, num_classes=num_classes, batch_norm=bn)
        if args.net == 'AlexNet':
            net = AlexNet(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet18':
            net = ResNet18(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet34':
            net = ResNet34(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet50':
            net = ResNet50(num_ch=num_ch, num_classes=num_classes)
        if args.net == 'ResNet101':
            net = ResNet101(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'LeNet':
        #     net = LeNet(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'GoogLeNet':
        #     net = GoogLeNet(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'MobileNetV2':
        #     net = MobileNetV2(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'DenseNet121':
        #     net = DenseNet121(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'EfficientNetB0':
        #     net = EfficientNetB0(num_ch=num_ch, num_classes=num_classes)
        # if args.net == 'DenseNetL2':
        #     net = DenseNetL2(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        # if args.net == 'DenseNetL4':
        #     net = DenseNetL4(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        # if args.net == 'DenseNetL6':
        #     net = DenseNetL6(num_ch=num_ch * imsize ** 2, num_classes=num_classes, h=args.width)
        # if args.net == 'MinCNN':
        #     net = MinCNN(num_ch=num_ch, num_classes=num_classes, h=args.width, fs=args.filter_size, ps=args.pooling_size)
    return net
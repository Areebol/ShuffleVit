'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-19 20:42:54
'''

def get_model(args):
    if args.model_name == 'vit':
        from model.vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    elif args.model_name == 'shuffle_vit':
        from model.vit_shuffle import ShuffleViT
        net = ShuffleViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            shuffle_num=args.shuffle_num,
            shuffle_channel=args.shuffle_channel,
            )
    elif args.model_name == 'resNet':
        from model.resnet import ResNet, ResidualBlock
        net = ResNet(
            ResidualBlock,
            [2,2,2])
    elif args.model_name == 'moe':
        from model.moe import Net
        net = Net()
    elif args.model_name == 'vit_moe':
        from model.dp_vit_moe import MoEViT
        net = MoEViT(            
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            num_experts=args.num_experts,
            ep_world_size=args.ep_world_size,
            top_k=args.top_k,
            min_capacity=args.min_capacity,
            noisy_gate_policy=args.noisy_gate_policy
            )
    # elif args.model_name == 'vit_moe':
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

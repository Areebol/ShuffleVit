'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-07-19 20:42:54
'''

# Return model


def get_model(args):
    # Set Vit model
    if args.model_name == 'vit':
        from model.vit import ViT
        net = ViT(
            args
        )
    # Set shuffle Vit model
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
    # Set resNet model
    elif args.model_name == 'resNet':
        from model.resnet import ResNet, ResidualBlock
        net = ResNet(
            ResidualBlock,
            [2, 2, 2])
    # Set moe Vit model based on tutel
    elif args.model_name == 'tutel_vit_moe':
        from model.tutel_vit_moe import MoEViT
        net = MoEViT(
            args
            # args.in_c,
            # args.num_classes,
            # img_size=args.size,
            # patch=args.patch,
            # dropout=args.dropout,
            # mlp_hidden=args.mlp_hidden,
            # num_layers=args.num_layers,
            # hidden=args.hidden,
            # head=args.head,
            # is_cls_token=args.is_cls_token,
            # num_experts=args.num_experts,
            # ep_world_size=args.ep_world_size,
            # top_k=args.top_k,
            # min_capacity=args.min_capacity,
            # noisy_gate_policy=args.noisy_gate_policy
        )
    # TODO Test dp moe model
    elif args.model_name == 'dp_vit_moe':
        from model.dp_vit_moe import MoEViT
        net = MoEVit(
            args
            # args.in_c,
            # args.num_classes,
            # img_size=args.size,
            # patch=args.patch,
            # dropout=args.dropout,
            # mlp_hidden=args.mlp_hidden,
            # num_layers=args.num_layers,
            # hidden=args.hidden,
            # head=args.head,
            # is_cls_token=args.is_cls_token,
            # num_experts=args.num_experts,
            # ep_world_size=args.ep_world_size,
            # top_k=args.top_k,
            # min_capacity=args.min_capacity,
            # noisy_gate_policy=args.noisy_gate_policy
        )
    # TODO Set moe Vit model based on deepspeed
    # elif args.model_name == 'dp_vit_moe':
    # Raise exception
    else:
        raise NotImplementedError(
            f"{args.model_name} is not implemented yet...")

    return net

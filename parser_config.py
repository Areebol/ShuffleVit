'''
Descripttion: 
version: 1.0
Author: Areebol
Date: 2023-09-07 19:54:24
'''
import argparse
# Add parserItem to parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="c10",
                    type=str, help="[c10, c100, svhn]")
parser.add_argument("--model-name", default="vit",
                    help="[vit,resNet,tutel_vit_moe,shuffle_vit]", type=str)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=150, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
parser.add_argument("--shuffle-num", default=1, type=int)
parser.add_argument("--shuffle-channel", default=64, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument('--T', default=1.0, type=float,
                    help='temperature scaling')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--moe',
                    default=False,
                    action='store_true',
                    help='use deepspeed mixture of experts (moe)')


parser.add_argument(
    '--moe-param-group',
    default=False,
    action='store_true',
    help='(moe) create separate moe param groups, required when using ZeRO w. MoE'
)

# Add model args
parser.add_argument('--in-c',
                    default=3,
                    type=int,
                    help='Set image input channel'
                    )
parser.add_argument("--num-classes",
                    default=10,
                    type=int,
                    help="Set classification number of classes"
                    )
parser.add_argument("--img-size",
                    default=32,
                    type=int,
                    help="Set input image's size")
parser.add_argument("--patch",
                    default=8,
                    type=int,
                    help="Set vision transformer's patch size")
parser.add_argument("--dropout",
                    default=0.0,
                    type=float,
                    help="Set dropout rate of the whole model")
parser.add_argument("--num-layers",
                    default=5,
                    type=int,
                    help="Set number of layers of the vision transformer")
parser.add_argument("--hidden",
                    default=384,
                    type=int,
                    help="Set input channel of the mlp inside the vit")
parser.add_argument("--mlp-hidden",
                    default=384,
                    type=int,
                    help="Set hidden channel of mlp inside the vit")
parser.add_argument("--head",
                    default=12,
                    type=int,
                    help="Set number of heads inside the vit")
parser.add_argument("--off-cls-token",
                    action="store_true",
                    help="Set whether to use cls_token")

# Add moe model args
parser.add_argument('--num-experts',
                    type=int,
                    nargs='+',
                    default=[
                        1,
                    ],
                    help='Set number of experts list, MoE related')
parser.add_argument('--ep-world-size',
                    default=1,
                    type=int,
                    help='Set expert parallel world size')
parser.add_argument('--top-k',
                    default=1,
                    type=int,
                    help='Set top-k gate policy (top 1 and 2 supported)')
parser.add_argument('--min-capacity',
                    default=0,
                    type=int,
                    help='Set (moe) minimum capacity of an expert regardless of the capacity_factor'
                    )
parser.add_argument('--noisy-gate-policy',
                    default=None,
                    type=str,
                    help='Set (moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
                    )

args = parser.parse_args()
args.benchmark = True if not args.off_benchmark else False
args.num_workers = 4
args.is_cls_token = True if not args.off_cls_token else False

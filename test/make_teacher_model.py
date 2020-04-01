import torch
import argparse

import models.cifar as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--arch", "-a", metavar="ARCH", default="resnet20",
                    choices=model_names,
                    help="model architecture: " +
                        " | ".join(model_names) +
                        " (default: resnet18)")
parser.add_argument("--depth", type=int, default=29, help="Model depth.")
parser.add_argument("--block-name", type=str, default="BasicBlock",
                    help="the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)")
parser.add_argument("--cardinality", type=int, default=8, help="Model cardinality (group).")
parser.add_argument("--widen-factor", type=int, default=4, help="Widen factor. 4 -> 64, 8 -> 128, ...")
parser.add_argument("--growthRate", type=int, default=12, help="Growth rate for DenseNet.")
parser.add_argument("--compressionRate", type=int, default=2, help="Compression Rate (theta) for DenseNet.")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--save-filepath", type=str)
parser.add_argument("--num-classes", type=int)
args = parser.parse_args()

if args.arch.startswith("resnext"):
    model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=args.num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.startswith("densenet"):
    model = models.__dict__[args.arch](
                num_classes=args.num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                dropRate=args.drop,
            )
elif args.arch.startswith("wrn"):
    model = models.__dict__[args.arch](
                num_classes=args.num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                dropRate=args.drop,
            )
elif args.arch.endswith("resnet"):
    model = models.__dict__[args.arch](
                num_classes=args.num_classes,
                depth=args.depth,
                block_name=args.block_name,
            )
else:
    model = models.__dict__[args.arch](num_classes=num_classes)
model = torch.nn.DataParallel(model)


checkpoint = torch.load(args.checkpoint)
print("got model with acc: {}".format(checkpoint["acc"]))
model.load_state_dict(checkpoint["state_dict"])
torch.save(model, args.save_filepath)


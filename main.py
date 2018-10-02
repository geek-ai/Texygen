from argparse import ArgumentParser
import sys

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


supported_gans = {
    'seqgan': Seqgan,
    'gsgan': Gsgan,
    'textgan': TextganMmd,
    'leakgan': Leakgan,
    'rankgan': Rankgan,
    'maligan': Maligan,
    'mle': Mle
}
supported_training = {'oracle', 'cfg', 'real'}


def set_gan(gan_name):
    try:
        Gan = supported_gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd():
    parser = ArgumentParser()
    parser.add_argument('-g', '--gan-type', help='The type of GAN to use',
                        choices=set(supported_gans.keys()), default='mle')
    parser.add_argument('-t', '--train-type', help='Type of training to use',
                        choices=supported_training, default='oracle')
    parser.add_argument('-d', '--data', default='data/image_coco.txt')
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unused_args = parse_cmd()
    gan = set_gan(args.gan_type)
    train_f = set_training(gan, args.train_type)
    if args.train_type == 'real':
        train_f(args.data)
    else:
        train_f()

import sys

from colorama import Fore
import tensorflow as tf

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
    flags = tf.app.flags
    flags.DEFINE_enum('gan_type', 'mle', list(supported_gans.keys()),
                      'Type of GAN to use')
    flags.DEFINE_enum('train_type', 'oracle', supported_training,
                      'Type of training to use')
    flags.DEFINE_string('data', 'data/image_coco.txt', '')
    return

if __name__ == '__main__':
    parse_cmd()
    tf.app.flags.DEFINE_string('oracle_file', 'save/oracle.txt', '')
    tf.app.flags.DEFINE_string('generator_file', 'save/generator.txt', '')
    tf.app.flags.DEFINE_string('test_file', 'save/test_file.txt', '')
    flags = tf.app.flags.FLAGS
    gan = set_gan(flags.gan_type)
    train_f = set_training(gan, flags.train_type)
    if flags.train_type == 'real':
        train_f(flags.data)
    else:
        train_f()

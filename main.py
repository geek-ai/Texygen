import getopt
import sys

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


def set_gan(gan_name):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    try:
        Gan = gans[gan_name.lower()]
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


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:t:")
        print(argv)
        print(opts)
        opt_arg = dict(opts)
        print(opt_arg)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                gan_func(opt_arg['-d'])
            else:
                gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])

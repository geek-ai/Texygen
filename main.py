from models.seqgan.Seqgan import Seqgan
from models.maligan_basic.Maligan import Maligan
from models.textGan_basic.Textgan import TextganBasic
from models.textGan_MMD.Textgan import TextganMmd
from models.pg_bleu.Pgbleu import Pgbleu
from models.rankgan.Rankgan import Rankgan

if __name__ == '__main__':
    # gans = []

    seqgan = Seqgan()
    seqgan.train_oracle()
    # maligan = Maligan()
    # # maligan.train_oracle()
    # textganbasic = TextganBasic()
    # # textganbasic.train_oracle()
    # textgan = TextganMmd()
    # # textgan.train_oracle()
    # pg = Pgbleu()
    # # pg.train_oracle()
    # rankgan = Rankgan()
    # # rankgan.train_oracle()
    # gans.append(seqgan)
    # gans.append(maligan)
    # gans.append(textganbasic)
    # gans.append(textgan)
    # gans.append(pg)
    # gans.append(rankgan)


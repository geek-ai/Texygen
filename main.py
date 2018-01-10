from models.seqgan.Seqgan import Seqgan

if __name__ == '__main__':
    # gans = []

    seqgan = Seqgan()
    # seqgan.train_oracle()
    seqgan.generate_num = 10000
    seqgan.train_real('data/shi.txt')
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


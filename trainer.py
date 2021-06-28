import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        if self.loader_train is None:
            pass
        elif len(self.loader_train) < 100:
            self.print_array = [len(self.loader_train)]
        else:
            self.print_array = [x * len(self.loader_train) // 10 for x in range(1,11,1)]


    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.4e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        self.loader_train.dataset.set_scale(0)

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr[0], 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) in self.print_array:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), 3)
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr[0], idx_scale)
                    sr = utility.quantize(sr, self.args.yuv_range)
                    if self.args.better_patch:
                        sr = utility.getBetterPatch(self.args.better_patch, sr, lr[1], hr)
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, 0] += utility.calc_psnr(
                        sr, hr, scale, self.args.yuv_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, 1] += utility.calc_psnr(
                        lr[1], hr, scale, self.args.yuv_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, 2] += utility.calc_psnr(
                        lr[0], hr, scale, self.args.yuv_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr[0][:,0:3,:,:], hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, 0] /= len(d)
                self.ckp.log[-1, idx_data, 1] /= len(d)
                self.ckp.log[-1, idx_data, 2] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} (VVC: {:.3f}, Input: {:.3f}) @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, 0],
                        best[0][idx_data, 1],
                        best[0][idx_data, 2],
                        best[1][idx_data, 0] + 1
                    )
                )
                self.ckp.writeCSVFile(best[0][idx_data, 0], self.ckp.log[-1, idx_data, 0],
                                      best[0][idx_data, 1], best[1][idx_data, idx_scale] + 1)

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) if not isinstance(a, list) else self.prepare(*a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


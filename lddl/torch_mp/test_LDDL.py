#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import argparse
import logging
import numpy as np
import os
import random
import time
import torch
from transformers import BertTokenizerFast
import lddl.torch
from lddl.torch_mp import get_bert_pretrain_data_loader
from lddl.torch_mp.utils import barrier, get_rank
from lddl.utils import mkdir

import warnings
warnings.filterwarnings("ignore")


def get_batch_seq_lens(attention_mask):
  return attention_mask.sum(dim=1).int()


class AverageMeter:
  """
  Computes and stores the average and current value
  """

  def __init__(self, warmup=0, keep=False):
    self.reset()
    self.warmup = warmup
    self.keep = keep

  def reset(self):
    self.val = 0
    self.avg = 0
    self.max = float('-inf')
    self.min = float('inf')
    self.sum = 0
    self.count = 0
    self.iters = 0
    self.vals = []

  def update(self, val, n=1):
    self.iters += 1
    self.val = val

    if self.iters > self.warmup:
      self.sum += val * n
      self.max = max(val, self.max)
      self.min = min(val, self.min)
      self.count += n
      self.avg = self.sum / self.count
      if self.keep:
        self.vals.append(val)


class Histogram:
  """
  Computes and stores the histogram of values.
  """

  def __init__(self):
    self.hist = np.zeros((1,), dtype=np.uint64)

  def update(self, val, n=1):
    if val >= self.hist.shape[0]:
      new_hist = np.zeros((val + 1,), dtype=np.uint64)
      new_hist[:self.hist.shape[0]] = self.hist[:]
      self.hist = new_hist
    self.hist[val] += n

  def update_with_tensor(self, t):
    for v in t.flatten().tolist():
      self.update(v)


def main(args):
  torch.cuda.set_device(args.local_rank)
  print(args.local_rank)
  world_size = int(os.getenv('WORLD_SIZE', 1))
  if world_size > 1:
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
  # group_ranks = [0,2]
  # group = torch.distributed.new_group(group_ranks)
  if get_rank() == 0 and args.seq_len_dir is not None:
    mkdir(args.seq_len_dir)
  samples_consumed_dploader = 0
  global_batch_size_on_this_data_parallel_rank = 16
  micro_batch_size = 4

  loader = get_bert_pretrain_data_loader(
        args.path,
        dp_rank=int(args.local_rank % 2),
        local_rank=args.local_rank,
        shuffle_buffer_size=16384,
        shuffle_buffer_warmup_factor=16,
        vocab_file=args.vocab_file,
        data_loader_kwargs={
            'batch_size': 2,
            'num_workers': 1,
            'prefetch_factor': 1
        },
        mlm_probability=0.15,
        base_seed=1234,
        log_level=logging.CRITICAL,
        log_dir="/tmp/log",
        return_raw_samples=False,
        start_epoch=0,
        sequence_length_alignment=8,
        ignore_index=-1,
        samples_seen = samples_consumed_dploader,
        micro_batch_size = 2
    )

  loader_og = lddl.torch.get_bert_pretrain_data_loader(
      args.path,
      local_rank=args.local_rank,
      shuffle_buffer_size=args.shuffle_buffer_size,
      shuffle_buffer_warmup_factor=args.shuffle_buffer_warmup_factor,
      vocab_file=args.vocab_file,
      data_loader_kwargs={
          'batch_size': 2,
          'num_workers': 1,
          'prefetch_factor': 1
      },
      mlm_probability=args.mlm_probability,
      base_seed=args.seed,
      log_dir=args.log_dir,
      log_level=logging.CRITICAL,
      return_raw_samples=args.debug,
      start_epoch=args.start_epoch,
      sequence_length_alignment=8,
      ignore_index=args.ignore_index,
    )

  if os.path.isfile(args.vocab_file):
    test_tokenizer = BertTokenizerFast(args.vocab_file)
  else:
    test_tokenizer = BertTokenizerFast.from_pretrained(args.vocab_file)

  # meter = AverageMeter(warmup=args.warmup)

  # lens_shape = (args.epochs, min(len(loader), args.iters_per_epoch))
  # min_lens, max_lens, batch_sizes, padded_lens = (
  #     np.zeros(lens_shape, dtype=np.uint16),
  #     np.zeros(lens_shape, dtype=np.uint16),
  #     np.zeros(lens_shape, dtype=np.uint16),
  #     np.zeros(lens_shape, dtype=np.uint16),
  # )
  # seq_len_hist = Histogram()
  # padded_zero_hist = Histogram()
  
  # print(len(loader))
  print(len(loader))
  print(len(loader_og))
  id = None
  for i, data in enumerate(loader):
    pass
    # id = data
    # print(data)
    # #if i == 1:
    # break
  #print(id['text'][0][:16])
    # if args.local_rank in group_ranks:
    #   ten_copy = data['text'].clone()
    #   ten = data['text'].cuda()
    #   torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.SUM, group=group)
    #   assert(torch.equal(2 * ten_copy,ten.cpu()))
    #print(data['text'].shape)
  # for i, data in enumerate(loader_og):
  #   if(torch.equal(data['input_ids'][0][:16], id['text'][0][:16])):
  #     print(data)
    # if i == 1:
    #   break
    

def attach_args(parser=argparse.ArgumentParser()):
  parser.add_argument('--path', type=str, required=True)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument('--warmup', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--iters-per-epoch', type=int, default=float('inf'))
  parser.add_argument('--prefetch', type=int, default=2)
  parser.add_argument(
      '--local_rank',
      type=int,
      default=os.getenv('LOCAL_RANK', 0),
  )
  parser.add_argument('--mlm-probability', type=float, default=0.15)
  parser.add_argument('--shuffle-buffer-size', type=int, default=16384)
  parser.add_argument('--shuffle-buffer-warmup-factor', type=int, default=16)
  parser.add_argument('--vocab-file', type=str, required=True)
  parser.add_argument('--seed', type=int, default=127)
  parser.add_argument('--start-epoch', type=int, default=0)
  parser.add_argument('--debug', action='store_true', default=False)
  parser.add_argument('--log-freq', type=int, default=1000)
  parser.add_argument('--log-dir', type=str, default=None)
  parser.add_argument(
      '--log-level',
      type=str,
      choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
      default='WARNING',
  )
  parser.add_argument('--seq-len-dir', type=str, default=None)
  parser.add_argument('--sequence-length-alignment', type=int, default=8)
  parser.add_argument('--ignore-index', type=int, default=-1)
  return parser


if __name__ == '__main__':
  main(attach_args().parse_args())
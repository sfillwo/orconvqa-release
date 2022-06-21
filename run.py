#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import argparse
import logging
import os
import random
import glob
import timeit
import json
import linecache
import faiss
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import (LazyQuacDatasetGlobal, RawResult, 
                   write_predictions, write_final_predictions, 
                   get_retrieval_metrics, gen_reader_features)
from retriever_utils import RetrieverDataset
from modeling import Pipeline, AlbertForRetrieverOnlyPositivePassage, BertForOrconvqaGlobal
from scorer import quac_eval


# In[2]:


logger = logging.getLogger(__name__)

ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
}


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
        orig_eval_file = args.orig_test_file
    else:
        eval_file = args.dev_file
        orig_eval_file = args.orig_dev_file

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, retriever_tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.retriever_query_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False, 
                           include_first_for_retriever=args.include_first_for_retriever)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    retriever_run_dict, rarank_run_dict = {}, {}
    examples, features = {}, {}
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        # print(qids)
        question_texts = np.asarray(
            batch['question_text']).reshape(-1).tolist()
        # answer_texts ("") and answer_starts (0) to dummy lists
        answer_texts = [""]*len(question_texts)
        answer_starts = [0]*len(question_texts)
        query_reps = gen_query_reps(args, model, batch)
        retrieval_results = retrieve(args, qids, query_reps,
                                     passage_ids, passage_id_to_idx,
                                     qrels,
                                     gpu_index, include_positive_passage=False)
        retriever_probs = retrieval_results['retriever_probs']
        # print('retriever_probs before', retriever_probs)
        pids_for_reader = retrieval_results['pids_for_reader']
        passages_for_reader = retrieval_results['passages_for_reader']
        # labels_for_reader = retrieval_results['labels_for_reader']

        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts, pids_for_reader,
                                                                           passages_for_reader,
                                                                           reader_tokenizer,
                                                                           args.reader_max_seq_length,
                                                                           is_training=False)
        example_ids = reader_batch['example_id']
        # print('example_ids', example_ids)
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(args.device)
                        for k, v in reader_batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {'input_ids':      reader_batch['input_ids'],
                      'attention_mask': reader_batch['input_mask'],
                      'token_type_ids': reader_batch['segment_ids']}
            outputs = model.reader(**inputs)
        
        retriever_probs = retriever_probs.reshape(-1).tolist()
        # print('retriever_probs after', retriever_probs)
        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]), 
                               retriever_prob=retriever_probs[i])
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    output_prediction_file = os.path.join(
        predict_dir, "instance_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        predict_dir, "instance_nbest_predictions_{}.json".format(prefix))
    output_final_prediction_file = os.path.join(
        predict_dir, "final_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            predict_dir, "instance_null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                        args.version_2_with_negative, args.null_score_diff_threshold)
    write_final_predictions(all_predictions, output_final_prediction_file, 
                            use_rerank_prob=args.use_rerank_prob, 
                            use_retriever_prob=args.use_retriever_prob)

    return all_predictions


# In[6]:


def gen_query_reps(args, model, batch):
    model.eval()
    batch = {k: v.to(args.device) for k, v in batch.items() 
             if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}
    with torch.no_grad():
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


# In[7]:


def retrieve(args, qids, query_reps,
             passage_ids, passage_id_to_idx,
             qrels,
             gpu_index, include_positive_passage=False):
    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    pids_for_retriever = passage_ids[pidx_for_retriever]
    # passage_reps_for_retriever = passage_reps[pidx_for_retriever]

    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]

    pids_for_reader = passage_ids[pidx_for_reader]
    passages_for_reader = get_passages(pidx_for_reader, args)

    return {'pidx_for_retriever': pidx_for_retriever,
            'pids_for_retriever': pids_for_retriever,
            # 'passage_reps_for_retriever': passage_reps_for_retriever,
            'retriever_probs': retriever_probs,
            'pidx_for_reader': pidx_for_reader,
            'pids_for_reader': pids_for_reader,
            'passages_for_reader': passages_for_reader}



def get_passage(i, args):
    line = linecache.getline(args.blocks_path, i + 1)
    line = json.loads(line.strip())
    return line['text']
get_passages = np.vectorize(get_passage)


parser = argparse.ArgumentParser()

# arguments shared by the retriever and reader

parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/train.txt',
                    type=str, required=False,
                    help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/dev.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/test.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/dev.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--orig_test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/quac/test.txt',
                    type=str, required=False,
                    help="original quac json for evaluation.")
parser.add_argument("--qrels", default='/mnt/scratch/chenqu/orconvqa/v5/retrieval/qrels.txt', type=str, required=False,
                    help="qrels to evaluate open retrieval")
# parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
#                     help="all blocks text")
parser.add_argument("--blocks_path", default='/mnt/scratch/chenqu/orconvqa/v3/all/all_blocks.txt', type=str, required=False,
                    help="all blocks text")
parser.add_argument("--passage_reps_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_reps.pkl',
                    type=str, required=False, help="passage representations")
parser.add_argument("--passage_ids_path", default='/mnt/scratch/chenqu/orconvqa/v5/passage_reps/combined/passage_ids.pkl',
                    type=str, required=False, help="passage ids")
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/orconvqa_output/release_test', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--load_small", default=True, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=2, type=int, required=False,
                    help="number of workers for dataloader")

parser.add_argument("--global_mode", default=True, type=str2bool, required=False,
                    help="maxmize the prob of the true answer given all passages")
parser.add_argument("--history_num", default=1, type=int, required=False,
                    help="number of history turns to use")
parser.add_argument("--prepend_history_questions", default=True, type=str2bool, required=False,
                    help="whether to prepend history questions to the current question")
parser.add_argument("--prepend_history_answers", default=False, type=str2bool, required=False,
                    help="whether to prepend history answers to the current question")

parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=True, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--best_global_step", default=40, type=int, required=False,
                    help="used when only do_test")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=1.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=1,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=20,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='',
                    help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='',
                    help="Can be used for distant debugging.")

# retriever arguments
parser.add_argument("--retriever_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--retriever_model_type", default='albert', type=str, required=False,
                    help="retriever model type")
parser.add_argument("--retriever_model_name_or_path", default='albert-base-v1', type=str, required=False,
                    help="retriever model name")
parser.add_argument("--retriever_tokenizer_name", default="albert-base-v1", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--retriever_cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/albert_v1/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--retrieve_checkpoint",
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_33/checkpoint-5917', type=str,
                    help="generate query/passage representations with this checkpoint")
parser.add_argument("--retrieve_tokenizer_dir",
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_33/', type=str,
                    help="dir that contains tokenizer files")

parser.add_argument("--given_query", default=True, type=str2bool,
                    help="Whether query is given.")
parser.add_argument("--given_passage", default=False, type=str2bool,
                    help="Whether passage is given. Passages are not given when jointly train")
parser.add_argument("--is_pretraining", default=False, type=str2bool,
                    help="Whether is pretraining. We fine tune the query encoder in retriever")
parser.add_argument("--include_first_for_retriever", default=True, type=str2bool,
                    help="include the first question in a dialog in addition to history_num for retriever (not reader)")
# parser.add_argument("--only_positive_passage", default=True, type=str2bool,
#                     help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
parser.add_argument("--retriever_query_max_seq_length", default=128, type=int,
                    help="The maximum input sequence length of query.")
parser.add_argument("--retriever_passage_max_seq_length", default=384, type=int,
                    help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--top_k_for_retriever", default=100, type=int,
                    help="retrieve top k passages for a query, these passages will be used to update the query encoder")
parser.add_argument("--use_retriever_prob", default=True, type=str2bool,
                    help="include albert retriever probs in final answer ranking")

# reader arguments
parser.add_argument("--reader_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--reader_model_name_or_path", default='bert-base-uncased', type=str, required=False,
                    help="reader model name")
parser.add_argument("--reader_model_type", default='bert', type=str, required=False,
                    help="reader model type")
parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--reader_cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--reader_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=384, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument('--version_2_with_negative', default=True, type=str2bool, required=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--reader_max_query_length", default=125, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                         "be truncated to this length.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=40, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                         "and end predictions are not conditioned on one another.")
parser.add_argument("--qa_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--retrieval_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--top_k_for_reader", default=5, type=int,
                    help="update the reader with top k passages")
parser.add_argument("--use_rerank_prob", default=True, type=str2bool,
                    help="include rerank probs in final answer ranking")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')
# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
# we now only support joint training on a single card
# we will request two cards, one for torch and the other one for faiss
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    # torch.cuda.set_device(0)
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()


model = Pipeline()

args.retriever_model_type = args.retriever_model_type.lower()
retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
retriever_config = retriever_config_class.from_pretrained(args.retrieve_checkpoint)

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
retriever_model = retriever_model_class.from_pretrained(args.retrieve_checkpoint, force_download=True)

model.retriever = retriever_model
# do not need and do not tune passage encoder
model.retriever.passage_encoder = None
model.retriever.passage_proj = None

args.reader_model_type = args.reader_model_type.lower()
reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
reader_config = reader_config_class.from_pretrained(args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
                                                    cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_config.num_qa_labels = 2
# this not used for BertForOrconvqaGlobal
reader_config.num_retrieval_labels = 2
reader_config.qa_loss_factor = args.qa_loss_factor
reader_config.retrieval_loss_factor = args.retrieval_loss_factor

reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                          do_lower_case=args.do_lower_case,
                                                          cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)
reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                  from_tf=bool(
                                                      '.ckpt' in args.reader_model_name_or_path),
                                                  config=reader_config,
                                                  cache_dir=args.reader_cache_dir if args.reader_cache_dir else None)

model.reader = reader_model

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

k = 1000
logger.info(f'loading passage ids from {args.passage_ids_path}')
with open(args.passage_ids_path, 'rb') as handle:
    passage_ids = json.load(handle)["ids"][:k]
    passage_ids = np.asarray(passage_ids)

logger.info(f'loading passage reps from {args.passage_reps_path}')
with open(args.passage_reps_path, 'rb') as handle:
    passage_reps = pkl.load(handle)[:k]


logger.info('constructing passage faiss_index')
if torch.cuda.is_available():
    faiss_res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(args.proj_size)
    index.add(passage_reps)
    gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, index)
else:
    index = faiss.IndexFlatIP(args.proj_size)
    index.add(passage_reps)
    gpu_index = index

del passage_reps


# logger.info(f'loading all blocks from {args.blocks_path}')
# with open(args.blocks_path, 'rb') as handle:
#     blocks_array = pkl.load(handle)


logger.info(f'loading qrels from {args.qrels}')
with open(args.qrels) as handle:
    qrels = json.load(handle)

passage_id_to_idx = {}
for i, pid in enumerate(passage_ids):
    passage_id_to_idx[pid] = i

#todo remove all things related to evaluation!!!


if args.do_test and args.local_rank in [-1, 0]:
    best_global_step = args.best_global_step
    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(
        args.retriever_tokenizer_dir, do_lower_case=args.do_lower_case)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_dir, do_lower_case=args.do_lower_case)
    best_checkpoint = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(best_global_step))
    logger.info("Test the best checkpoint: %s", best_checkpoint)

    model = Pipeline()
    model.retriever = retriever_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'retriever'), force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.reader = reader_model_class.from_pretrained(
        os.path.join(best_checkpoint, 'reader'), force_download=True)
    model.to(args.device)

    # Evaluate
    result = evaluate(args, model, retriever_tokenizer,
                      reader_tokenizer, prefix='test')

    test_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'test_metrics.json')
    with open(test_metrics_file, 'w') as fout:
        json.dump(result, fout)

    logger.info("Test Result: {}".format(result))


# In[ ]:





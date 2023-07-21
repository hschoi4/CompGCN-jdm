from helper import *
from tqdm import tqdm, trange
from data_loader import *
import wandb
# from carbontracker.tracker import CarbonTracker
import numpy as np
from collections import Counter


# sys.path.append('./')
from model.models import *


class Runner(object):

    def load_data(self):
        """
            Reading in raw triples and converts it into a standard format.

            Parameters
            ----------
            self.p.dataset:         Takes in the name of the dataset (FB15k-237)

            Returns
            -------
            self.ent2id:            Entity to unique identifier mapping
            self.id2rel:            Inverse mapping of self.ent2id
            self.rel2id:            Relation to unique identifier mapping
            self.num_ent:           Number of entities in the Knowledge graph
            self.num_rel:           Number of relations in the Knowledge graph
            self.embed_dim:         Embedding dimension used
            self.data['train']:     Stores the triples corresponding to training dataset
            self.data['valid']:     Stores the triples corresponding to validation dataset
            self.data['test']:      Stores the triples corresponding to test dataset
            self.data_iter:			The dataloader for different data splits

        """
        custom_data = self.p.dataset == "RezoJDM16k"

        if not custom_data:
            ent_set, rel_set = OrderedSet(), OrderedSet()
            for split in ['train', 'test', 'valid']:
                for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)

            self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

        else:
            # CustomData: RezoJDM16k 

            self.ent2id = {}
            with open('./data/{}/{}.txt'.format(self.p.dataset, 'entities')) as f:
                for line in f.readlines():
                    tokens = line.strip().split()
                    _id = int(tokens.pop(0))
                    _ent = ' '.join(tokens)
                    self.ent2id[_ent] = _id

            self.rel2id = {}
            with open('./data/{}/{}.txt'.format(self.p.dataset, 'relations')) as f:
                for line in f.readlines():
                    tokens = line.strip().split()
                    _id = int(tokens.pop(0))
                    _rel = ' '.join(tokens)
                    self.rel2id[_rel] = _id

        # TODO: 50 years in the future come back and take into account existing reverse relations ,
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for rel, idx, in self.rel2id.items()})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                if custom_data:
                    sub, rel, obj = map(int, line.strip().split('\t'))
                else:
                    # Map things from node names to node IDs
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        # TRIM CODE COMES HERE
        if self.p.trim:
            '''
                1. actually do the trim of train set -> select correct triples from valid_head
                2. re make sr2o (dont know what it deos but can remake)
                3. update p.num_ent; num_rel truc
            '''
            tr = self.data['train']
            vl = self.data['valid']
            trim_ratio = self.p.trim_ratio

            trn = np.array(tr)
            trn_trimmed = trn[np.random.choice(trn.shape[0], size=int(len(tr)*trim_ratio), replace=False), :]

            # Count entities in train and how many times they occur
            ent_counter = Counter(trn_trimmed[:, 0])
            ent_counter.update(trn_trimmed[:, 2])

            rel_counter = Counter(trn_trimmed[:, 1])

            # which val entities are not in train
            vl_transductive = []
            for s,r,o in vl:
                if s not in ent_counter or o not in ent_counter or r not in rel_counter:
                    continue
                vl_transductive.append((s,r,o))

            # Its possibel that vl_transductive has less triples than we need (based on trim ratio)
            # if so, throw error asking for some change maube bigger ratio
            if len(vl_transductive) < int(len(vl)*trim_ratio):
                raise ValueError('During trimming; not enough valid triples left. Retry with bigger trim ratio.')

            # we trim from this
            vln_transductive = np.array(vl_transductive)
            vln_trimmed = vln_transductive[np.random.choice(vln_transductive.shape[0], size=int(len(vl)*trim_ratio), replace=False), :]

            # project hail mary: we dont know whats happening we only know that this just might fucking work
            sr2o = ddict(set)
            for sub, rel, obj in trn_trimmed:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

            # Lets update self.data now again
            self.data['train'] = trn_trimmed.tolist()
            self.data['valid'] = vln_trimmed.tolist()
            self.data['test'] = vln_trimmed.tolist()

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
            Constructor of the runner class

            Parameters
            ----------

            Returns
            -------
            Constructs the adjacency matrix for GCN
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index  = torch.LongTensor(edge_index).to(self.device).t()
        edge_type   = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
            Constructor of the runner class

            Parameters
            ----------
            params:         List of hyperparameters of the model

            Returns
            -------
            Creates computational graph and optimizer

        """

        self.p      = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model     = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

        if self.p.use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="gnn-jdm-compgcn",
                # track hyperparameters and run metadata
                config=vars(self.p),
                name=self.p.name
            )

    def load_entity_vectors(self):
        ''' Go to the dataset folder based on p.dataset and get the vectors torch file.'''
        vectorspath = './data/{}/{}'.format(self.p.dataset, 'vectors_entity_fasttext.torch')
        try:
            vectors = torch.load(vectorspath)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {vectorspath} not found. Run the corresponding script in ./scripts subdirectory.")
        return vectors

    def add_model(self, model, score_func):
        """
			Creates the computational graph

			Parameters
			----------
			model:     Contains the model name to be created
			score_func: Contains the score function (decoder) we're using

			Returns
			-------
			Creates the computational graph for model and initializes it

		"""

        model_name = '{}_{}'.format(model, score_func)

        if self.p.use_fasttext:
            vectors = self.load_entity_vectors()
            self.logger.info(f"Used FastText vectors. Shape: {vectors.shape}")
        else:
            vectors = None

        if   model_name.lower() == 'compgcn_transe':    model = CompGCN_TransE(self.edge_index, self.edge_type, vectors=vectors, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':  model = CompGCN_DistMult(self.edge_index, self.edge_type, vectors=vectors, params=self.p)
        elif model_name.lower() == 'compgcn_conve':     model = CompGCN_ConvE(self.edge_index, self.edge_type, vectors=vectors, params=self.p)
        else: raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        """
            Creates an optimizer for training the parameters

            Parameters
            ----------
            parameters:         The parameters of the model

            Returns
            -------
            Returns an optimizer for learning the parameters of the model

        """

        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
            Function to read a batch of data and move the tensors in batch to CPU/GPU

            Parameters
            ----------
            batch: 		the batch to process
            split: (string) If split == 'train', 'valid' or 'test' split


            Returns
            -------
            Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
            Function to save a model. It saves the model parameters, best validation scores,
            best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

            Parameters
            ----------
            save_path: path where the model is saved

            Returns
            -------
        """

        state = {
            'state_dict' : self.model.state_dict(),
            'best_val'   : self.best_val,
            'best_epoch' : self.best_epoch,
            'optimizer'  : self.optimizer.state_dict(),
            'args'       : vars(self.p)
        }
        torch.save(state, save_path)

    # noinspection PyAttributeOutsideInit
    def load_model(self, load_path):
        """
            Function to load a saved model

            Parameters
            ----------
            load_path: path to the saved model

            Returns
            -------
        """

        state             = torch.load(load_path, map_location=self.device)
        state_dict        = state['state_dict']
        self.best_val     = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
            Function to evaluate the model on validation or test set

            Parameters
            ----------
            split: (string) If split == 'valid' then evaluate on the validation set, else the test set
            epoch: (int) Current epoch count

            Returns
            -------
            resutls:			The evaluation results containing the following:
                results['mr']:         	Average of ranks_left and ranks_right
                results['mrr']:         Mean Reciprocal Rank
                results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        left_results  = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results       = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))

        if self.p.use_wandb:
            wandb.log(data={
                'mrr': results['mrr'],
                'left_mrr': results['left_mrr'],
                'right_mrr': results['right_mrr'],
                'mr': results['mr'],
                'left_mr': results['left_mr'],
                'right_mr': results['right_mr'],
                'hits@1': results['hits@1'],
                'left_hits@1': results['left_hits@1'],
                'right_hits@1': results['right_hits@1'],
                'hits@3': results['hits@3'],
                'left_hits@3': results['left_hits@3'],
                'right_hits@3': results['right_hits@3'],
                'hits@10': results['hits@10'],
                'left_hits@10': results['left_hits@10'],
                'right_hits@10': results['right_hits@10'],
            }, step=epoch)
        return results

    def predict(self, split='valid', mode='tail_batch', report_all:bool = True):
        """
            Function to run model evaluation for a given mode

            Parameters
            ----------
            split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
            mode: (string):		Can be 'head_batch' or 'tail_batch'
            report_all: (bool): If true, we return individual metric scores as well

            Returns
            -------
            resutls:			The evaluation results containing the following:
                results['mr']:         	Average of ranks_left and ranks_right
                results['mrr']:         Mean Reciprocal Rank
                results['hits@k']:      Probability of getting the correct prediction in top-k ranks based on predicted score

        """
        self.model.eval()

        if report_all:
            all_ranks = torch.Tensor()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label    = self.read_batch(batch, split)
                pred            = self.model.forward(sub, rel)
                b_range         = torch.arange(pred.size()[0], device=self.device)
                target_pred     = pred[b_range, obj]
                pred            = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj]  = target_pred
                ranks           = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

                ranks           = ranks.float()

                if report_all:
                    all_ranks = torch.cat([all_ranks, ranks.clone().detach().cpu()], dim=0)

                results['count']    = torch.numel(ranks)        + results.get('count', 0.0)
                results['mr']       = torch.sum(ranks).item()   + results.get('mr',    0.0)
                results['mrr']      = torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        if report_all:
            return results, all_ranks

        return results


    def run_epoch(self, epoch, val_mrr = 0):
        """
            Function to run one epoch of training

            Parameters
            ----------
            epoch: current epoch count
            val_mrr: TODO

            Returns
            -------
            loss: The loss value after the completion of one epoch
        """

        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred    = self.model.forward(sub, rel)
            loss    = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        if self.p.use_wandb:
            wandb.log(data={'loss': loss}, step=epoch)
        return loss

    def fit(self):
        """
            Function to run training and evaluation of model

            Parameters
            ----------

            Returns
            -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0

        # tracker = CarbonTracker(epochs=self.p.max_epochs, log_dir=self.p.log_dir, log_file_prefix="carbontracker")

        for epoch in range(self.p.max_epochs):
            # tracker.epoch_start()

            # train_loss  = self.run_epoch(epoch, val_mrr)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val      = val_results
                self.best_val_mrr  = val_results['mrr']
                self.best_epoch    = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 25:
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

            # tracker.epoch_end()

        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch)
        pprint(test_results)

        # tracker.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name',        default='testrun',                  help='Set run name for saving/restoring models')
    parser.add_argument('-data',        dest='dataset',         default='RezoJDM16k',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',       dest='model',       default='compgcn',      help='Model Name')
    parser.add_argument('-score_func',  dest='score_func',  default='conve',        help='Score Function for Link prediction')
    parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-use_wandb', type=str2bool, nargs='?', const=True, default=False
                        , help='Set True for logging this exp on WandB')

    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gamma',       type=float,             default=40.0,           help='Margin')
    parser.add_argument('-gpu',     type=str,               default='0',            help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',       dest='max_epochs',  type=int,       default=500,    help='Number of epochs')
    parser.add_argument('-l2',      type=float,             default=0.0,            help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',      type=float,             default=0.001,          help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',  type=float,     default=0.1,    help='Label Smoothing')
    parser.add_argument('-num_workers', type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,       help='Seed for randomization')

    parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-num_bases',   dest='num_bases',   default=-1,     type=int,   help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',    dest='init_dim',    default=100,    type=int,   help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',     dest='gcn_dim',     default=200,    type=int,   help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',   dest='embed_dim',   default=None,   type=int,   help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',   dest='gcn_layer',   default=1,      type=int,   help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',    dest='dropout',     default=0.1,    type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',    dest='hid_drop',    default=0.3,    type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',   dest='hid_drop2',   default=0.3,    type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop',   dest='feat_drop',   default=0.3,    type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',     dest='k_w',         default=10,     type=int,   help='ConvE: k_w')
    parser.add_argument('-k_h',     dest='k_h',         default=20,     type=int,   help='ConvE: k_h')
    parser.add_argument('-num_filt',    dest='num_filt',    default=200,    type=int,   help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',      dest='ker_sz',      default=7,      type=int,   help='ConvE: Kernel size to use')

    parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')

    # Trimming parameters
    parser.add_argument('-trim', type=str2bool, nargs='?', const=True, default=False
                        , help='Set True to only do the exp on a very small subset. Ratio defaults to 0.05. Change trim_ratio si tu veux')
    parser.add_argument('-trim_ratio', dest='trim_ratio',     default=0.05,    type=float, help='Trim ratio to cut the train and valid sets')
    parser.add_argument('-use_fasttext', type=str2bool, nargs='?', const=True, default=False,
    help='If True, we use FastText vectors for entities. If they dont exist, run the script in ./scripts folder for your dataset')
    args = parser.parse_args()

    if not args.restore and '_' not in args.name:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    if args.use_fasttext and args.init_dim != 100:
        raise ValueError('When using FastText always have 100 dimensions for init ie nodes.')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Runner(args)
    model.fit()

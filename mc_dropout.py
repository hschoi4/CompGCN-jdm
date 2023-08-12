"""
# MC DropOut

Under regular classification MC Dropout works like this:

You take a **trained** model $f(\ , \theta^*)$ , and input some data to it (do inference).
But you add some noise (in the form of dropout) and do repeated inference.

#### Dropout

Dropout makes some parameters of the model zero so some neurons don't work. This is done randomly.
So let the model be defined by a specific set of $M$ parameters $\theta^t \in \mathcal{R}^M$.

Implementing dropout can be defined as randomly sampling a mask from a binary space of the same dimensions:
    $m \in \{ 0,1 \}^M$.

We can get a new set of parameters of the model by doing a hadamard product between the parameters and the mask by
$\theta_d^t = \theta^t \odot m$

#### Implications

In doing so, we effectively make a new model.
If we sample multiple masks, we get different models.

#### MC Dropout

Use different models (by different dropouts) to predictions on the same input. Use them to compute confidence intervals


## MCD on Link Prediction

We're can't do the _exact_ same thing in our task so we'll make some adjustments to it.

# TODO: write out the rest

Get a matrix of nument x 100
A .8 softmax score of 95% means a particular entity gets a softmax of 0.8 95% of the times or more

"""
import random
import time
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
# noinspection PyPackageRequirements
from mytorch.utils.goodies import FancyDict
from tqdm.auto import trange

from run import Runner


def enable_dropout(model: torch.nn.Module):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_vanilla_pred(model: Runner,
                     sub: Union[str, int],
                     rel: Union[str, int]
                     ):
    if type(sub) != type(rel):
        raise TypeError(f"Both subject (type: {type(sub)}) and relation (type: {type(rel)})"
                        " should be of the same type")

    if type(sub) == str:
        sub = model.ent2id[sub]
        sub = [sub, sub]
        rel = model.rel2id[rel]
        rel = [rel, rel]
    elif type(sub) == int:
        sub = [sub, sub]
        rel = [rel, rel]

    # Now we're ready
    t_sub = torch.tensor(sub, dtype=torch.int32, device=model.device)
    t_rel = torch.tensor(rel, dtype=torch.int32, device=model.device)
    model.model.eval()
    with torch.no_grad():
        op = model.model.forward(t_sub, t_rel)[0].detach().cpu()
    return op


# For simplicity's sake we write a cold for loop (no batching) to get results
# noinspection PyTypeChecker
def get_predictions(model: Runner,
                    sub: Union[str, int, list],
                    rel: Union[str, int, list],
                    #                     objs: List[List[Union[int, str]]] = None,
                    n: int = 100):
    """
        **sub**, and **rel** can be a str, an int, a list of str, or a list of int:
            'soleil i.a'
            1
            ['soleil i.a', 'lune i.a']
            [1, 2]

        If a list, we treat this as a batch.
        If a single value, we make it into a batch of 2

        ~~**objs** must be a list (but can contain strings or ints)~~
            ~~If not provided, we find the based on get_gold~~

        Recommended to not change n during one run. Bad things will happen.
    """
    if type(sub) != type(rel):
        raise TypeError(f"Both subject (type: {type(sub)}) and relation (type: {type(rel)})"
                        " should be of the same type")

    if type(sub) == str:
        sub = model.ent2id[sub]
        sub = [sub, sub]
        rel = model.rel2id[rel]
        rel = [rel, rel]
        singular = True
    elif type(sub) == int:
        sub = [sub, sub]
        rel = [rel, rel]
        singular = True
    elif type(sub) == list:
        if len(sub) != len(rel):
            raise ValueError(f"Both subject (len: {len(sub)}) and relation (len: {len(rel)})"
                             " should be of the same length")
        for i in range(len(sub)):
            sub[i] = sub[i] if type(sub[i]) is int else model.ent2id[sub[i]]
            rel[i] = rel[i] if type(rel[i]) is int else model.rel2id[rel[i]]
        singular = False
    else:
        raise TypeError(f'Unknown type for sub: {type(sub)}.')

    # Now we're ready
    t_sub = torch.tensor(sub, dtype=torch.int32, device=model.device)
    t_rel = torch.tensor(rel, dtype=torch.int32, device=model.device)

    if singular:
        pred = torch.zeros(2, n, model.p.num_ent, dtype=torch.float32, device=model.device)
    else:
        pred = torch.zeros(len(sub), n, model.p.num_ent, dtype=torch.float32, device=model.device)
    model.model.eval()
    enable_dropout(model.model)

    with torch.no_grad():
        for i in trange(n, desc="Actual Prediction", leave=False):
            pred[:, i, :] = model.model.forward(t_sub, t_rel)

    return pred if not singular else pred[0]


# noinspection PyTypeChecker
def get_predictions_dum(*args, **kwargs):
    # time.sleep(1)

    for _ in trange(3, desc="Actual Prediction", leave=False):
        time.sleep(0.2)
    return torch.randn(100, 26558, dtype=torch.float32)


# noinspection PyTypeChecker
def run(
        datasetname: str = 'RLF/lf-cp',
        checkpointname: str = './checkpoints/rlf-lf-cp_dropout',
        n_samples: int = 100,
        save_every: int = 100,
        gpu: str = '0'
):
    args = {'name': 'testrun',
            'dataset': datasetname,
            'model': 'compgcn',
            'score_func': 'conve',
            'opn': 'corr',
            'use_wandb': False,
            'batch_size': 128,
            'gamma': 40.0,
            'gpu': gpu,
            'max_epochs': 1,
            'l2': 0.0,
            'lr': 0.001,
            'lbl_smooth': 0.1,
            'num_workers': 10,
            'seed': 41504,
            'restore': False,
            'bias': False,
            'num_bases': -1,
            'init_dim': 100,
            'gcn_dim': 200,
            'embed_dim': None,
            'gcn_layer': 1,
            'dropout': 0.05,
            'hid_drop': 0.15,
            'hid_drop2': 0.15,
            'feat_drop': 0.15,
            'k_w': 10,
            'k_h': 20,
            'num_filt': 200,
            'ker_sz': 7,
            'log_dir': './log/',
            'config_dir': './config/',
            'trim': False,
            'trim_ratio': 0.00005,
            'use_fasttext': False
            }
    args = FancyDict(args)

    model = Runner(args)
    # # Now load the saved model
    model.load_model(checkpointname)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Step 1: Try to load dataframes for this dataset
    pathdir = Path('./mc_dropout_new') / Path(checkpointname).name
    try:
        df = pd.read_pickle(pathdir / 'graph.pickle')
    except FileNotFoundError:
        # Create the folder in case that doesn't exist either
        pathdir.mkdir(parents=True, exist_ok=True)

        df_tr = pd.DataFrame(model.data['train'], columns=['sub', 'rel', 'obj'])
        df_vl = pd.DataFrame(model.data['valid'], columns=['sub', 'rel', 'obj'])
        df_ts = pd.DataFrame(model.data['test'], columns=['sub', 'rel', 'obj'])

        tr_groups = df_tr.groupby(['sub', 'rel'])['obj'].apply(set)
        tr_groups = tr_groups.reset_index(name='train_objs')
        vl_groups = df_vl.groupby(['sub', 'rel'])['obj'].apply(set)
        vl_groups = vl_groups.reset_index(name='valid_objs')
        ts_groups = df_ts.groupby(['sub', 'rel'])['obj'].apply(set)
        ts_groups = ts_groups.reset_index(name='test_objs')

        def row_union(x: pd.Series) -> list:
            res = \
                set().union(x['train_objs'] if x['train_objs'] != -1 else set()
                            ).union(x['valid_objs'] if x['valid_objs'] != -1 else set()
                                    ).union(x['test_objs'] if x['test_objs'] != -1 else set())
            return list(res)

        df = pd.merge(left=tr_groups, right=vl_groups, how='outer', on=['sub', 'rel'])
        df = pd.merge(left=df, right=ts_groups, how='outer', on=['sub', 'rel'])
        df = df.fillna(-1)
        df['all_obj'] = df.apply(row_union, axis=1)
        df['index'] = df.index

        # Save this to disk
        df.to_pickle(pathdir / 'graph.pickle')

    # Now, we batch every {save_every} sub, rel combinations.
    # We process one sub, rel at once and don't use batches.
    # Why? Because this will be the most random way to do it

    # First find how many batches have already been generated
    try:
        last_stored = max(int(filename.name.split('.')[0]) for filename in pathdir.rglob('*.torch'))
    except ValueError:
        last_stored = 0
        print('Found no stored files. Will start from the start')

    for batch_i in trange(last_stored, df.shape[0], save_every, desc=f"Main: {last_stored} onwards:"):
        # Save {batchsize} predictions (no dropout; default way)
        # Save {batchsize}*{saveevery} MC Dropout predictions
        mcd_preds: Optional[torch.Tensor] = None  # We use this var to keep appending predictions to.
        vanilla_preds = []

        for i in trange(batch_i, batch_i + save_every, desc="In one batch", leave=False):
            try:
                sub, rel = df['sub'][i], df['rel'][i]
            except (KeyError, IndexError) as e:
                # we must be over the df limit
                break

            mcd_pred = get_predictions(model, int(sub), int(rel), n=n_samples)  # (n, num_ent)
            vanilla_preds.append(get_vanilla_pred(model, int(sub), int(rel)))
            if mcd_preds is not None:
                mcd_preds = torch.cat((mcd_preds, mcd_pred.unsqueeze(0)))  # (i, n, num_ent)
                del mcd_pred
            else:
                mcd_preds = mcd_pred.unsqueeze(0)

        # Dump this to disk
        torch.save({
            "mc_dropout": mcd_preds,
            "vanilla": torch.vstack(vanilla_preds)
        }, pathdir / f"{batch_i + save_every}.torch")
        del mcd_preds, vanilla_preds


if __name__ == '__main__':
    run(save_every=100)

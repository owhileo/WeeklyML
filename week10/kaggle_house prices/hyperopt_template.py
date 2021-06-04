from hyperopt import hp,STATUS_OK,Trials,fmin,tpe
from matplotlib import pyplot as plt
import numpy as np


# plt.switch_backend('agg')

best=0
def hyperopt_test(hyperopt_train_test,space,fig_path='hyperopt_test.png',max_evals=300):
    """

    def hyperopt_train_test(params):
        rf=RandomForestClassifier(**params)
        ...
        return score

    space = {
        'maxDepth': hp.choice('maxDepth', range(1,20)),
        'maxBins': hp.choice('maxBins', range(8,100,8)),
        'numTrees': hp.choice('numTrees', range(1,20)),
        'impurity': hp.choice('impurity', ["gini", "entropy"]),
        'subsamplingRate': hp.choice('subsamplingRate', np.arange(1,0,-0.05)),
    }
    """
    space_=dict([(x,hp.choice(x,y)) for x,y in space.items()])
    global best
    best = -1e14
    def f(params):
        global best
        acc = hyperopt_train_test(params)
        if acc > best:
            best = acc
            print('new best:', best, params)
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, space_, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print('best:',best)

    parameters=list(space_.keys())
    rows=2
    columns=(len(parameters)+1)//2
    f, axes = plt.subplots(nrows=rows,ncols=columns, figsize=(4*columns,5*rows))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [t['result']['loss'] for t in trials.trials]
        ys = np.array(ys)
        axes[i//columns][i%columns].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, cmap=cmap(float(i)/len(parameters)))
        axes[i//columns][i%columns].set_xticks(range(len(space[val])))
        if type(space[val][0])!=str :
            temp=np.round(space[val],decimals=1)
        else:
            temp=[x[:9] for x in space[val]]
        n_=len(temp)//15
        if n_>0:
            temp_=["",]*len(temp)
            for ii in range(0,len(temp),n_+1):
                temp_[ii]=temp[ii]
            temp=temp_
        axes[i//columns][i%columns].set_xticklabels(temp,rotation=-35,fontsize = 'small')
        axes[i//columns][i%columns].set_ylim([29250,30750])
        axes[i//columns][i%columns].set_title(val)

    plt.savefig(fig_path)

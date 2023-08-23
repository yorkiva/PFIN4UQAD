This repo is set up to use Evidential Deep Learning for jet classification using the `PFIN` model.
Setup the required environment and run the following command-

```
python train.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir 		Output directory for trained model
  --outdictdir          Output directory for trained model metadata
  --Np                  Number of constituents
  --NPhiI 		Number of hidden layer nodes for Interaction Phi
  --x-mode 		Mode of Interaction: ['sum', 'cat']
  --Phi-nodes           Comma-separated list of hidden layer nodes for Phi
  --F-nodes		Comma-separated list of hidden layer nodes for F
  --epochs		Epochs
  --label		a label for the model
  --batch-size          batch_size
  --data-loc 		Directory for data
  --data-type 		Dataset to train on
  --preload             Preload weights and biases from a pre-trained Model
  --preload-file        Location of the model to the preload
  --KLcoef 		annealing coefficient, nominal implies an epoch-dependent coefficient
  --mass-range          thresholds for jet mass range in the form 'LOGIC:M1,M2' where LOGIC can be AND (between the range) or OR (between the range)
  --pt-range 		thresholds for jet pT range in the form 'LOGIC:PT1,PT2' where LOGIC can be AND (between the range) or OR (outside the range)
  --eta-range 		thresholds for jet eta range in the form 'LOGIC:ETA1,ETA2' where LOGIC can be AND (between the range) or OR (outside the range)
  --skip-labels         Jet labels to drop from training
  --batch-mode          Set this flag when running in batch mode to suppress tqdm progress bars
  --use-softmax         Set this flag when using softmax probabilites
  --use-dropout         Set this flag when using dropout layers
```

To train for the `JetClass` dataset with multiple GPUs, use the script `train_mod.py`. This script is known to work on multiple GPUs on a single node. In addition to the options allowed for the `train.py` script, it also has a few extra options:

```
  --ndata-per-gpu       Only for jetclass data- number of data files per gpu (1 file = 1M jets
  --nodes 		Number of nodes (do not change it yet, default set to 1)
  --gpus GPUS           number of gpus per node
  --local-rank          ranking within the nodes (yet not to change)
  --IP IP               IP Address of Master Node
  --seed SEED           Random seed
```


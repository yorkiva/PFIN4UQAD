To evaluate your model, run the following command-

```
python evaluate_model.py <options>
```

The available options are:

```
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory for evaluation results
  --data-type {topdata,jetclass,JNqgmerged,jetnet}
                        Dataset to evaluate on
  --make-file           Set this flag when recording evaluation results
  --data-loc DATA_LOC   Directory for data
  --modeldir MODELDIR   Directory for trained model parameters
  --modeldictdir MODELDICTDIR
                        Directory for trained model metadata
  --tag TAG             Optional tag to only store results of certain models with tag in the name
  --type {ensemble,edl,dropout}
                        Type of model to evaluate
  --batch-mode          Set this flag when running in batch mode to suppress tqdm progress bars
```

Import stubs so the authors' PAD model classes (S4A-Models `ae5a376`) can be
instantiated and their checkpoints loaded for inference without installing
pytorch-lightning 1.6 or tensorboardX into `deep_field`. Only inference is run;
nothing that the stubs replace (training loop, logging, hyperparameter saving)
affects a forward pass. Put this folder first on `sys.path`.

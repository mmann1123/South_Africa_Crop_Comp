"""Re-run CNN-BiLSTM and TempCNN pixel inference from existing checkpoints with
outputs redirected to experiments/paired_bootstrap/rerun/ (out_of_sample untouched)."""
import os, sys, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "out_of_sample"))
for modname, out in [("inference_cnn_bilstm", "predictions_cnn_bilstm_rerun.csv"),
                     ("inference_tempcnn", "predictions_tempcnn_rerun.csv")]:
    mod = importlib.import_module(modname)
    mod.OUTPUT_CSV = os.path.join(HERE, "rerun", out)
    mod.main()

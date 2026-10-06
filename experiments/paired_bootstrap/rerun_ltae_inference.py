"""Re-run L-TAE pixel inference from the existing checkpoints WITHOUT touching
out_of_sample/: outputs are redirected to experiments/paired_bootstrap/rerun/.
Used to check whether the published L-TAE (pixel) row is reproducible."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "out_of_sample"))
import inference_ltae as mod
mod.OUTPUT_CSV = os.path.join(HERE, "rerun", "predictions_ltae_rerun.csv")
mod.main()

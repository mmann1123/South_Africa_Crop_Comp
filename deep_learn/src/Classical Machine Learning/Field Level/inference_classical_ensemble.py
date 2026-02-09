import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import MODEL_DIR

import pandas as pd
from joblib import load

TEST_PARQUET        = os.path.join(MODEL_DIR, "test_data.parquet")
VOTING_PIPELINE     = os.path.join(MODEL_DIR, "ensemble_voting.pkl")
STACK_PIPELINE      = os.path.join(MODEL_DIR, "ensemble_stacking.pkl")
LABEL_ENCODER_FILE  = os.path.join(MODEL_DIR, "label_encoder.pkl")

OUTPUT_STACK = os.path.join(MODEL_DIR, "results_field_level_stacking_test.csv")
OUTPUT_VOTE  = os.path.join(MODEL_DIR, "results_field_level_voting_test.csv")


def main():
    
    df_test = pd.read_parquet(TEST_PARQUET)
    print("Loaded test split:", df_test.shape)
    fids        = df_test['fid'].to_numpy()
    true_labels = df_test['crop_name'].to_numpy()
    X_test      = df_test.drop(columns=['fid', 'crop_name'])
    voting_pipe = load(VOTING_PIPELINE)
    stacking_pipe = load(STACK_PIPELINE)
    le = load(LABEL_ENCODER_FILE)
    print("Loaded pipelines and LabelEncoder.")
    codes_stack = stacking_pipe.predict(X_test)
    codes_vote  = voting_pipe.predict(X_test)
    preds_stack = le.inverse_transform(codes_stack)
    preds_vote  = le.inverse_transform(codes_vote)

    df_stack = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_stack
    })
    df_stack.to_csv(OUTPUT_STACK, index=False)
    print(f"Saved stacking results → {OUTPUT_STACK}")

    df_vote = pd.DataFrame({
        'fid':             fids,
        'true_label':      true_labels,
        'predicted_label': preds_vote
    })
    df_vote.to_csv(OUTPUT_VOTE, index=False)
    print(f"Saved voting results  → {OUTPUT_VOTE}")

if __name__ == '__main__':
    main()

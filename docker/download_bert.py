import os
from transformers import BertModel, BertTokenizer

out_dir = os.environ.get("BERT_PATH", "/workspace/data/bert-base-uncased")
os.makedirs(out_dir, exist_ok=True)


# Download and save into the repo-expected directory
tok = BertTokenizer.from_pretrained("bert-base-uncased")
mdl = BertModel.from_pretrained("bert-base-uncased")
tok.save_pretrained(out_dir)
mdl.save_pretrained(out_dir)
print(f"Saved BERT-base-uncased to: {out_dir}")
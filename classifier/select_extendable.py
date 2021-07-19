from pathlib import Path


current_dir = Path(__file__).resolve().parent
dat_file = current_dir / "responses_ann_new.txt"
out_dat = current_dir / "extendable_qa.csv"

with open(dat_file, mode='r', encoding='utf-8') as f, \
        open(out_dat, mode='w', encoding='utf-8', newline='') as out_f:
    extendable = [line.split(';')[0] for line in f if ';1' in line]
    for qa_pair in extendable:
        out_f.write(f'{qa_pair}\n')

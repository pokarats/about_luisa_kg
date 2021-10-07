import itertools
from statistics import mean
from timeit import default_timer as timer
from datetime import timedelta
from bleurt import score
import argparse
from pathlib import Path


def cl_parser(argv=None):
    """
    Parse command line arguments

    :param argv:
    :return:
    """
    parser = argparse.ArgumentParser(description="Options for running BLEURT")
    parser.add_argument("--reference",
                        default="outputs/gold_extension.txt",
                        type=str,
                        help="Path to gold/reference file",
                        )
    parser.add_argument("--candidate",
                        default="outputs/tf_idf.txt",
                        type=str,
                        help="Path to candidate file",
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="auto_eval_scores",
                        help="Where to store automatic evaluation files")

    parser.add_argument("--scores_file",
                        type=str,
                        help="File name for eval score file")
    parser.add_argument("--checkpoint",
                        default="bleurt-base-128",
                        type=str,
                        help="Which BLEURT model checkpoint to use?")
    parser.add_argument("--queries",
                        type=bool,
                        default=False,
                        help="True = file contents are queries; False = answers")

    return parser.parse_args(argv)


def main():
    args = cl_parser()
    print(args)

    checkpoint = f"bleurt/{args.checkpoint}"
    ref_path = Path(args.reference)
    cand_path = Path(args.candidate)

    references = []
    candidates = []

    assert ref_path.exists(), f"Reference file {args.reference} not found!"
    assert cand_path.exists(), f"Candidate file {args.reference} not found!"

    with open(ref_path, mode="r", encoding="utf-8") as ref_file:
        with open(cand_path, mode="r", encoding="utf-8") as cand_file:
            for ref_sentence, cand_sentence in itertools.zip_longest(
                    ref_file, cand_file, fillvalue=None):
                assert ref_sentence is not None, (
                    "Reference sentence not found, are you sure that the files have "
                    "the same size?")
                assert cand_sentence is not None, (
                    "Candidate sentence not found, are you sure that the files have "
                    "the same size?")
                references.append(ref_sentence.rstrip())
                candidates.append(cand_sentence.rstrip())

    assert len(references) == len(candidates)

    start_time = timer()
    # load BleurtScorer
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    assert type(scores) == list and len(scores) == len(references)
    end_time = timer()
    bleurt_time = timedelta(seconds=end_time-start_time)
    print(f"Scoring finished in: {bleurt_time}")

    # print average score to standard output and write to file
    scores_dir = Path(__file__).resolve().parent / args.output_dir
    try:
        scores_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{scores_dir} exists. Scores file will be saved there.")

    if args.scores_file is not None:
        scores_path = scores_dir / args.scores_file
    else:
        scores_path = scores_dir / f"auto_eval_{cand_path.stem}"

    total_score = sum(scores)
    mean_score = mean(scores)

    with open(scores_path, mode="w", encoding="utf-8") as scores_f:
        for sc in scores:
            print(f"{sc}", file=scores_f)
        print(f"Total: {total_score:.2f}\n"
              f"Average: {mean_score:.2f}\n"
              f"Scoring time: {bleurt_time}", file=scores_f)

    print(f"Total: {total_score:.2f}\n"
          f"Average: {mean_score:.2f}")


if __name__ == '__main__':
    main()

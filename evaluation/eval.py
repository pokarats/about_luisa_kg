import argparse
from argparse import Namespace
import random
from pathlib import Path


def cl_parser(argv=None):
    """
    Parse command line arguments

    :param argv:
    :return:
    """
    parser = argparse.ArgumentParser(description="Evaluation Preprocessing")
    parser.add_argument("--seed",
                        default=35,
                        type=int,
                        help="Random generator seed for reproducibility")
    parser.add_argument("--num_samples",
                        default=3,
                        type=int,
                        help="Number of samples per set")
    parser.add_argument("--file_path",
                        default="sample_queries_file.txt",
                        type=str,
                        help="Path to file",
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="human_eval_files",
                        help="Where to store evaluation sample indices files")
    # A: 1=gold, 2=base, 3=paraphrased
    # B: 1=base, 2=paraphrased, 3=gold
    # C: 1=paraphrased, 2=gold, 3=base
    parser.add_argument("--version",
                        type=str,
                        help="Which survey version? A|B|C")
    parser.add_argument("--queries",
                        type=bool,
                        default=False,
                        help="True = file contents are queries; False = answers")

    return parser.parse_args(argv)


def lines_from_file(path):
    """
    Yield line from file path with trailing whitespaces removed

    :param path: path to file
    :return: each line with trailing whitespaces removed
    """
    with open(path) as f:
        for line in f:
            yield line.rstrip()


def write_to_file(list_of_lines, path, delimiter="\n"):
    """
    Write list or iterable of str to file path, with specified delimiter

    :param delimiter:
    :param list_of_lines: list/iterable of str
    :param path: out file path
    :return:
    """
    with open(path, mode="w", encoding="utf-8") as out_file:
        for line in list_of_lines:
            out_file.write(f"{line}{delimiter}")


class Generated:
    def __init__(self, args=None):
        self._args = args
        self.texts = [generated_line for generated_line in lines_from_file(self._args.file_path)]
        self.set_one_indices = self._get_indices(self._args.num_samples, 1)
        self.set_two_indices = self._get_indices(self._args.num_samples, 2)
        self.set_three_indices = self._get_indices(self._args.num_samples, 3)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.texts[item]
        if isinstance(item, list):
            return [self.texts[sample] for sample in item]

        raise TypeError("Invalid Argument Type!!")

    def _get_indices(self, num_samples, set_id):
        """
        Get the indices for a stored file or sample from the indices of the file lines.

        :param num_samples:
        :param set_id:
        :return:
        """
        idx_file_path = Path(__file__).resolve().parent / f"{str(set_id)}.txt"
        try:
            return [int(idx) for idx in lines_from_file(idx_file_path)]
        except FileNotFoundError:
            return self.sample_indices(num_samples, set_id)

    def sample_indices(self, num_samples=10, set_id=None):
        """
        Sample specified number of indices/elements for a specified set_id.
        If set_id is None, sample all 3 sets

        :param num_samples: number of samples for each group or set
        :param set_id: 1 | 2 | 3
        :return:
        """
        idx_file_dir = Path(__file__).resolve().parent
        random.seed(self._args.seed)
        indices = list(range(len(self.texts)))
        random.shuffle(indices)

        if set_id is None:
            # create indices for all 3 sets for the number of samples/set specified
            self.set_one_indices = indices[-num_samples:]
            idx_file_path = idx_file_dir / "1.txt"
            write_to_file(self.set_one_indices, idx_file_path)
            del indices[-num_samples:]

            self.set_two_indices = indices[-num_samples:]
            idx_file_path = idx_file_dir / "2.txt"
            write_to_file(self.set_two_indices, idx_file_path)
            del indices[-num_samples:]

            self.set_three_indices = indices[-num_samples:]
            idx_file_path = idx_file_dir / "3.txt"
            write_to_file(self.set_three_indices, idx_file_path)
            del indices[-num_samples:]
        else:
            # if specific set is specified, create indices for that set
            # set_id order: 1 = last set of 10, 2 = second to last, 3 = third to last set of 10
            assert 0 < int(set_id) <= 3

            del_idx = 0
            while del_idx < int(set_id) - 1:
                del indices[-num_samples:]
                del_idx += 1

            sampled_idx = indices[-num_samples:]
            idx_file_path = idx_file_dir / f"{set_id}.txt"
            write_to_file(sampled_idx, idx_file_path)
            return sampled_idx

    @staticmethod
    def shuffle_indices(*args, seed=35):
        all_items = [element for element_list in args for element in element_list]
        random.seed(seed)
        random.shuffle(all_items)

        return all_items

    @classmethod
    def get_generated_extensions(cls, another_args, indices):
        extensions = Generated(another_args)
        return extensions[indices]

    def make_shuffled_samples(self, *args):
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / f"{self._args.output_dir}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{out_dir} already exists! Files will be saved there!")

        out_filepath = out_dir / f"shuffled_queries.txt" if self._args.queries \
            else out_dir / f"shuffled_answers{self._args.version}.txt"

        items = self.shuffle_indices(*args, seed=self._args.seed)
        write_to_file(items, out_filepath)


def main():
    args = cl_parser()
    print(args)

    samples = Generated(args)
    print(f"number of queries: {len(samples)}")
    print(f"indices of set 1: {samples.set_one_indices}")
    print(f"indices of set 2: {samples.set_two_indices}")
    print(f"indices of set 3: {samples.set_three_indices}")

    print(f"set 1 samples: {samples[samples.set_one_indices]}")

    print(f"all eval indices shuffled: {Generated.shuffle_indices(samples.set_one_indices, samples.set_two_indices, samples.set_three_indices)}")

    args.version = "A"
    if args.version.upper() == "A":
        # A: 1=gold, 2=base, 3=paraphrased
        another_args = Namespace(**vars(args))
        another_args.file_path = "sample_answer_file_gold.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        another_args.file_path = "sample_answer_file_base.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        another_args.file_path = "sample_answer_file_para.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        samples.make_shuffled_samples(gold_extensions, base_extensions, para_extensions)

    args.version = "B"
    if args.version.upper() == "B":
        # B: 1=base, 2=paraphrased, 3=gold
        another_args = Namespace(**vars(args))
        another_args.file_path = "sample_answer_file_gold.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        another_args.file_path = "sample_answer_file_base.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        another_args.file_path = "sample_answer_file_para.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        samples.make_shuffled_samples(base_extensions, para_extensions, gold_extensions)

    args.version = "C"
    if args.version.upper() == "C":
        # C: 1=paraphrased, 2=gold, 3=base
        another_args = Namespace(**vars(args))
        another_args.file_path = "sample_answer_file_gold.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        another_args.file_path = "sample_answer_file_base.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        another_args.file_path = "sample_answer_file_para.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        samples.make_shuffled_samples(para_extensions, gold_extensions, base_extensions)


if __name__ == '__main__':
    main()

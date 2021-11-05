from evaluation.eval import cl_parser, Generated, write_to_file
from argparse import Namespace


def main():
    args = cl_parser()
    print(args)
    samples = Generated(args)
    print(f"number of queries: {len(samples)}")
    print(f"indices of set 1: {samples.set_one_indices}")
    print(f"indices of set 2: {samples.set_two_indices}")
    print(f"indices of set 3: {samples.set_three_indices}")

    if args.queries:
        shuffled_idx = Generated.shuffle_indices(samples.set_one_indices,
                                                 samples.set_two_indices,
                                                 samples.set_three_indices)
        write_to_file(shuffled_idx, "evaluation/shuffled_idx.txt")
        one = samples[samples.set_one_indices]
        two = samples[samples.set_two_indices]
        three = samples[samples.set_three_indices]
        samples.make_shuffled_samples(one, two, three)

    if args.version.upper() == "A":
        # A: 1=gold, 2=base, 3=paraphrased
        another_args = Namespace(**vars(args))
        another_args.file_path = "outputs/gold_extension.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        another_args.file_path = "outputs/tf_idf.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        another_args.file_path = "outputs/t5_paraphrase.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        samples.make_shuffled_samples(gold_extensions, base_extensions, para_extensions)

    if args.version.upper() == "B":
        # B: 1=base, 2=paraphrased, 3=gold
        another_args = Namespace(**vars(args))
        another_args.file_path = "outputs/gold_extension.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        another_args.file_path = "outputs/tf_idf.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        another_args.file_path = "outputs/t5_paraphrase.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        samples.make_shuffled_samples(base_extensions, para_extensions, gold_extensions)

    if args.version.upper() == "C":
        # C: 1=paraphrased, 2=gold, 3=base
        another_args = Namespace(**vars(args))
        another_args.file_path = "outputs/gold_extension.txt"
        gold_extensions = Generated.get_generated_extensions(another_args, samples.set_two_indices)

        another_args.file_path = "outputs/tf_idf.txt"
        base_extensions = Generated.get_generated_extensions(another_args, samples.set_three_indices)

        another_args.file_path = "outputs/t5_paraphrase.txt"
        para_extensions = Generated.get_generated_extensions(another_args, samples.set_one_indices)

        samples.make_shuffled_samples(para_extensions, gold_extensions, base_extensions)


if __name__ == '__main__':
    main()

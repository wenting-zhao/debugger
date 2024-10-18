import argparse
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model to use")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset to use")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("-n", type=int, default=1)
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=512)
    llm = LLM(model=args.model_name)
    outputs = llm.generate(ds["prompt"], sampling_params)

    results = []
    for output in outputs:
        generated_texts = [one.text for one in output.outputs]
        results.append(generated_texts)
        print(generated_texts)
    ds = ds.add_column(name="prediction", column=results)
    out_name = args.dataset_name.split("/")[-1]
    out_name = f"{out_name}_predictions.json"
    ds.to_json(out_name)

if __name__ == '__main__':
    main()

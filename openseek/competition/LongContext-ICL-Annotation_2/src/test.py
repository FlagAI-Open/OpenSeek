import argparse
import openai


TASK_ID = "test"

client = openai.OpenAI(
	api_key="EMPTY",
	base_url="http://localhost:2026/v1",
)


def call_llm(prompt_text: str) -> str:
	"""Call the local OpenAI-compatible model endpoint."""
	try:
		response = client.chat.completions.create(
			model="/root/ascend/log/debug/Qwen3-4B",
			messages=[
				{
					"role": "system",
					"content": "You are a Senior GPU Kernel Engineer specializing in Triton and PyTorch autograd.",
				},
				{"role": "user", "content": prompt_text},
			],
			temperature=0.6,
			top_p=0.85,
			max_tokens=5000,
		)
		whole_result = response.choices[0].message.content or ""
		print(f"\n[Model Response] Task {TASK_ID}:\n{whole_result}\n" + "=" * 50)
		return whole_result
	except Exception as e:
		print(f"Exception occurred for Task {TASK_ID}: {e}")
		return ""


def main() -> None:
	prompt = "You are a Triton programming expert. Function: Tensor element-wise division.div(input, other, *, rounding_mode=None, out=None) -> Tensor.\nOutput runnable code"
	output = call_llm(prompt)
	if not output:
		raise SystemExit(1)


if __name__ == "__main__":
	main()

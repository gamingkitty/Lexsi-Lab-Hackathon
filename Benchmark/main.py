import auditkit as ak
import ollama
import polars as pl
import time
import jev_scorer as jev
import json
from pathlib import Path

# df = pl.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/data/harmful-behaviors.csv")
df_jailbreak = pl.read_csv(
    "https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/resolve/main/JailBreakV_28K/JailBreakV_28K.csv"
)

def get_aya_response(prompt):
    response = ollama.chat(
        model='hf.co/CohereLabs/tiny-aya-global-GGUF:Q4_K_M',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    )
    return response.message.content

def get_jev_probabilities(prompt, question, choices):
    response = ollama.systemone(
        model="d3:4b-q8_0",
        state=prompt,
        questions={
            "decision": {
                "type": "choice",
                "instructions": question,
                "criteria": choices,
            }
        },
    )

    return response.answers["decision"]["probabilities"]


def main():
    batch_size = 100
    checkpoint_path = Path("jev_checkpoint.json")
    rows = df_jailbreak.head(5000)
    total = rows.height

    jev.configure_jev(
        judge_model="d3:4b-q8_0",
        results_path="jev_results.jsonl",
    )

    config = ak.RunConfig(
        judge_model=jev.JUDGE_MODEL,
        judge_prompt_version=jev.JUDGE_PROMPT_VERSION,
    )

    # Resume after the last successfully completed batch.
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        start_index = checkpoint["next_index"]
    else:
        start_index = 0
        ak.DiskCache().clear()

    if start_index >= total:
        print(f"All {total} samples are already completed.")
        return

    print(f"Starting at sample {start_index + 1} of {total}.")
    t0 = time.perf_counter()

    for batch_start in range(start_index, total, batch_size):
        batch_end = min(batch_start + batch_size, total)

        samples = [
            ak.Sample(
                input=row["jailbreak_query"],
                target=row["redteam_query"],
            )
            for row in rows.slice(
                batch_start, batch_end - batch_start
            ).iter_rows(named=True)
        ]

        print(f"\nEvaluating samples {batch_start + 1}–{batch_end}...")
        batch_t0 = time.perf_counter()

        result = ak.evaluate(
            samples,
            model=lambda prompts: [
                get_aya_response(prompt) for prompt in prompts
            ],
            scorers=[jev.jev_metric],
            config=config,
        )

        # Advance only after evaluation returns successfully.
        # Atomic replacement protects against a partially written checkpoint.
        temporary_path = checkpoint_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps({"next_index": batch_end}, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)

        print(f"Batch took: {time.perf_counter() - batch_t0:.1f} seconds")
        print(result.summary())
        print(f"Checkpoint saved: {batch_end}/{total} completed.")

    print(f"\nThis session took: {time.perf_counter() - t0:.1f} seconds")


if __name__ == "__main__":
    main()

import re
import json
import os
import time
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model


# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"


def load_alpaca_data() -> List[Dict[str, str]]:

    dataset = []
    with open("./data/alpaca_eval_first_30.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    return dataset

def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """
    Construct a prompt for an LLM judge to evaluate two model responses.

    Args:
        query: the question given to the two models (from AlpacaEval)
        response_E: output from model E on query
        response_F: output from model F on query
    Returns:
        Prompt for the LLM judge.
    
    Consider: The judge is an LLM that will output free-form text. How will you 
    design the prompt so that you can reliably determine which response it preferred?
    Your llm_judge_template and extract_llm_judge_preference should work together.
    """
    # TODO complete for question 3b
    prompt = f"""You will be given two responses from different models to the same user query.
Compare them objectively and comprehensively, then select the better one.

Query: {query}

response_E: {response_E}

response_F: {response_F}

Now report your result. The LAST line of your response must be your final choice, formatted exactly like this:

#### {MODEL_E_PREFERED_TAG}

or 

#### {MODEL_F_PREFERED_TAG}

For example, if you think responce_E is better, the last line must read exactly:

#### {MODEL_E_PREFERED_TAG}
""".strip()

    return prompt

def llm_judge_template_position_swaped(query: str, response_F: str, response_E: str) -> str:
    """
    Construct a prompt for an LLM judge to evaluate two model responses.

    Args:
        query: the question given to the two models (from AlpacaEval)
        response_E: output from model E on query
        response_F: output from model F on query
    Returns:
        Prompt for the LLM judge.
    
    Consider: The judge is an LLM that will output free-form text. How will you 
    design the prompt so that you can reliably determine which response it preferred?
    Your llm_judge_template and extract_llm_judge_preference should work together.
    """
    # TODO complete for question 3b
    prompt = f"""You will be given two responses from different models to the same user query.
Compare them objectively and comprehensively, then select the better one.

Query: {query}

response_F: {response_F}

response_E: {response_E}

Now report your result. The LAST line of your response must be your final choice, formatted exactly like this:

#### {MODEL_F_PREFERED_TAG}

or 

#### {MODEL_E_PREFERED_TAG}

For example, if you think responce_F is better, the last line must read exactly:

#### {MODEL_F_PREFERED_TAG}
""".strip()

    return prompt

def extract_llm_judge_preference(judge_output: str) -> str:
    """
    Extract the judge's preference from its output.

    Args:
        judge_output: the string sampled from the LLM judge.
    Returns:
        A string representing which response the judge preferred.
    
    This function should work in tandem with your llm_judge_template design.
    What if the judge's output is malformed or ambiguous?
    """
    # TODO complete for question 3b
    ANS_RE = re.compile(r"#### (<MODEL_[EF]_BETTER>)")
    matches = ANS_RE.findall(judge_output)

    if matches:
        # Take the last tag: the judge may restate the options while reasoning
        # before committing to a choice on its final line.
        match_str: str = matches[-1].strip()
        return match_str
    else:
        return NO_PREFERENCE_FOUND_TAG

RESULTS_PATH = "llm_judge_results.jsonl"
RESULTS_2_PATH = "llm_judge_results_2.jsonl"

def run_llm_judge_eval(out_path: str = RESULTS_2_PATH):
    """
    Run the LLM-as-a-judge evaluation comparing models E and F on AlpacaEval data.
    Use model Z as the judge.

    For each AlpacaEval instruction, you'll need responses from both models E and F,
    then have the judge compare them.

    Remember to save your results (model responses + judge outputs) - you will
    need them for Parts C and D.
    """
    # TODO complete for question 3b
    E_wins = 0
    F_wins = 0
    invalid = 0
    total_cost = .0
    dataset = load_alpaca_data()
    instruction_nums = len(dataset)

    with open(out_path, "w", encoding="utf-8") as out:
        for i, data_line in enumerate(tqdm(dataset, desc="judging")):
            instruction = data_line["instruction"]
            query = Query(turns=[
                    {"user": instruction}
                ])
            # Sampled once per instruction, outside the order loop: both orders
            # must judge the same pair of responses, or the comparison between
            # them mixes the position effect with fresh sampling noise.
            response_E = query_model(
                model_id="E",
                query=query
            )
            response_F = query_model(
                model_id="F",
                query=query
            )

            for j in range(2):
                if j % 2 == 0:
                    judge_prompt = llm_judge_template(instruction, response_E.text, response_F.text)
                else:
                    judge_prompt = llm_judge_template_position_swaped(instruction, response_F.text, response_E.text)

                judge_query = Query(turns=[
                    {"user": judge_prompt}
                ])
                judge_response = query_model(
                    model_id="Z",
                    query=judge_query
                )
                preference = extract_llm_judge_preference(judge_response.text)

                record = {
                    "index": i,
                    # 0 = model E shown first, 1 = model F shown first. Parts C
                    # and D need this to separate the two presentation orders.
                    "order": j,
                    "instruction": instruction,
                    "response_E": response_E.text,
                    "response_F": response_F.text,
                    "judge_response": judge_response.text,
                    "preference": preference,
                    "E_output_tokens": response_E.output_tokens,
                    "F_output_tokens": response_F.output_tokens,
                    "cost": response_E.cost + response_F.cost + judge_response.cost,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                if preference == MODEL_E_PREFERED_TAG:
                    E_wins += 1
                elif preference == MODEL_F_PREFERED_TAG:
                    F_wins += 1
                elif preference == NO_PREFERENCE_FOUND_TAG:
                    invalid += 1

                total_cost += record["cost"]

    # Each instruction is judged twice, once per presentation order, so the
    # denominator is the number of comparisons rather than of instructions.
    comparison_nums = E_wins + F_wins + invalid
    E_wins_rate = E_wins * 1.0 / comparison_nums
    F_wins_rate = F_wins * 1.0 / comparison_nums

    print(f"Result: \n")
    print(f"{instruction_nums} instructions, {comparison_nums} comparisons IN TOTAL: \n")
    print(f"Response_E wins: {E_wins}, accounting for {E_wins_rate}\n")
    print(f"Response_F wins: {F_wins}, accounting for {F_wins_rate}\n")
    print(f"Invalid preference nums: {invalid}")
    print(f"Cost: {total_cost}\n")
    print(f"Saved {comparison_nums} records to {out_path}")
    
    
def load_results(*paths: str) -> List[Dict]:
    """
    Load one or more saved evaluation files produced by run_llm_judge_eval.

    Pooling both presentation orders (E first and F first) is what cancels the
    judge's position bias, so parts C and D read them together.
    """
    records = []
    for file_index, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Result files written before the "order" field existed hold
                    # one presentation order each, so the file identifies it.
                    record.setdefault("order", file_index)
                    records.append(record)
    return records

def split_lengths_by_preference(records: List[Dict]) -> tuple:
    """
    Split output lengths into the ones the judge preferred and the ones it did not.

    Args:
        records: records loaded by load_results.
    Returns:
        (preferred_lengths, not_preferred_lengths), measured in output tokens.
    """
    preferred, not_preferred = [], []
    for r in records:
        if r["preference"] == MODEL_E_PREFERED_TAG:
            preferred.append(r["E_output_tokens"])
            not_preferred.append(r["F_output_tokens"])
        elif r["preference"] == MODEL_F_PREFERED_TAG:
            preferred.append(r["F_output_tokens"])
            not_preferred.append(r["E_output_tokens"])
        # Records with no extractable preference cannot be assigned to either
        # group and are dropped; report how many if this ever fires.
    return preferred, not_preferred

def plot_win_rates(
    paths: tuple = (RESULTS_PATH, RESULTS_2_PATH),
    out_path: str = "../../a4_tex/images/judge_win_rates.png",
) -> None:
    """
    For Part B: bar chart of the judge's win rates for models E and F.

    Shows each presentation order separately alongside the pooled result, so the
    headline win rate is never read without the ordering it depends on. Orders
    are taken from each record's "order" field, which load_results fills in from
    the source file for results written before that field existed.
    """
    records = load_results(*paths)

    groups = []
    for order in sorted({r["order"] for r in records}):
        recs = [r for r in records if r["order"] == order]
        e = sum(1 for r in recs if r["preference"] == MODEL_E_PREFERED_TAG)
        groups.append((len(recs), e))

    groups.append((sum(n for n, _ in groups), sum(e for _, e in groups)))

    labels = ["Model E shown\nfirst", "Model F shown\nfirst", "Pooled\n(counterbalanced)"]
    e_rates = [e / n * 100 for n, e in groups]
    f_rates = [100 - r for r in e_rates]

    x = range(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar([i - w / 2 for i in x], e_rates, width=w, color="#2a78d6",
                label="Model E", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], f_rates, width=w, color="#eb6834",
                label="Model F", zorder=3)

    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win rate (%)", color="#52514e")
    ax.set_title("Judge win rate by presentation order", color="#0b0b0b",
                 fontsize=11, pad=12)

    # 50% is the reference a position-indifferent judge would produce.
    ax.axhline(50, color="#c3c2b7", linewidth=1, linestyle="--", zorder=1)

    ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, labelcolor="#52514e", loc="upper right")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    color="#0b0b0b", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)

    for label, (n, e) in zip(labels, groups):
        print(f"{label.replace(chr(10), ' '):32} E {e}/{n} = {e/n:.1%}")
    print(f"Saved figure to {out_path}")

def plot_model_output_lengths(
    paths: tuple = (RESULTS_PATH, RESULTS_2_PATH),
    out_path: str = "../../a4_tex/images/judge_length_bias.png",
) -> None:
    """
    For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
    """
    # TODO complete for question 3d
    records = load_results(*paths)
    preferred, not_preferred = split_lengths_by_preference(records)
    dropped = len(records) - len(preferred)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    top = max(max(preferred), max(not_preferred))
    bins = [i * top / 10 for i in range(11)]

    # Grouped (side-by-side) bars rather than translucent overlap: blended fills
    # would create a third colour that belongs to neither group.
    ax.hist([preferred, not_preferred], bins=bins,
            color=["#2a78d6", "#eb6834"],
            label=["Preferred by judge", "Not preferred"], zorder=3)

    ax.set_xlabel("Response length (output tokens)", color="#52514e")
    ax.set_ylabel("Number of responses", color="#52514e")
    ax.set_title(f"Judge-preferred vs non-preferred response lengths "
                 f"(n={len(preferred)} each)", color="#0b0b0b", fontsize=10.5, pad=12)

    ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, labelcolor="#52514e")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)

    import statistics as stats
    print(f"Pooled {len(records)} records from {len(paths)} file(s); "
          f"{dropped} dropped for having no extractable preference.")
    print(f"Preferred     mean {stats.mean(preferred):7.0f}  median {stats.median(preferred):6.0f}")
    print(f"Not preferred mean {stats.mean(not_preferred):7.0f}  median {stats.median(not_preferred):6.0f}")
    print(f"Saved figure to {out_path}")

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    run_llm_judge_eval()
    #plot_model_output_lengths()

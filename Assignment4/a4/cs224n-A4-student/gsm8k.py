import re
import json
from typing import Dict
import os

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #



def eval_model_on_gsm8k() -> list:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.
    
    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer' fields.
    
    Think about: What metric will you use to evaluate performance? How will you 
    handle cases where the model's output cannot be parsed?
    """
    # TODO complete for question 2bi
    result = []
    with open("data/gsm8k_first_100.jsonl", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    for model_id in ["A", "B"]:
        correct = 0
        invalid = 0
        question_num = len(data)
        cost = .0
        for data_line in tqdm(data, desc=f"Model{model_id}"):
            question = data_line["question"]
            answer = data_line["numerical_answer"]

            prompt = standard_prompt_template(question)
            query = Query(turns=[
                {"user": prompt} 
            ])
            response = query_model(
                model_id=model_id,
                query=query
            )
            output = standard_output_extractor(response.text)

            if output != INVALID_ANS:
                try: 
                    if float(output) == float(answer):
                        correct += 1
                except ValueError:
                    invalid += 1
            else:
                invalid += 1

            cost += response.cost

        accuracy = correct * 1.0 / question_num
        false = question_num - correct - invalid

        print(f"Model {model_id} Result: \n")
        print(f"Correct answer nums: {correct}\n")
        print(f"Invalid answer nums: {invalid}\n")
        print(f"False answer nums: {false}\n")
        print(f"Accuracy: {accuracy}\n")
        print(f"Cost: {cost}\n")

        result.append({"model_id": model_id, "correct": correct, "invalid": invalid, "false": false, "accuracy": accuracy, "cost": cost})
    return result
              
    



def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.
    
    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.
    
    Look at standard_prompt_template() to understand the baseline approach. What 
    aspects of how you prompt the model might affect its reasoning or accuracy?
    
    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    # TODO complete for question 2bii

    prompt = f"""Solve the following problem using as many steps of reasoning as you need,
showing each calculation. Before you commit to an answer, check your work carefully.

Problem: {question}

Now report your result. The LAST line of your response must be the final numerical
answer, formatted exactly like this:

#### <numerical_answer>

Rules for that last line:
- It must begin with the four characters #### followed by a space.
- Put only a plain number after "####" -- no LaTeX, no \\boxed{{}}, no dollar signs,
  no units, no commas, no bold markers.
- Even if you already stated the answer during your reasoning, you must still repeat
  it on this final #### line. A bare number on its own is not acceptable.
- Write nothing at all after that line.

For example, if the answer is 42, the last line must read exactly:

#### 42
""".strip()

    return prompt

def eval_model_on_gsm8k_with_improved_prompt() -> list:
    """
    Evaluate model A using your superior_prompt_template.
    """
    # TODO complete for question 2bii
    result = []
    with open("data/gsm8k_first_100.jsonl", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    for model_id in ["A"]:
        correct = 0
        invalid = 0
        question_num = len(data)
        cost = .0
        for data_line in tqdm(data, desc=f"Model{model_id}"):
            question = data_line["question"]
            answer = data_line["numerical_answer"]

            prompt = superior_prompt_template(question)
            query = Query(turns=[
                {"user": prompt} 
            ])
            response = query_model(
                model_id=model_id,
                query=query
            )
            output = standard_output_extractor(response.text)

            if output != INVALID_ANS:
                try: 
                    if float(output) == float(answer):
                        correct += 1
                except ValueError:
                    invalid += 1
            else:
                invalid += 1

            cost += response.cost

        accuracy = correct * 1.0 / question_num
        false = question_num - correct - invalid

        print(f"Model {model_id} Result: \n")
        print(f"Correct answer nums: {correct}\n")
        print(f"Invalid answer nums: {invalid}\n")
        print(f"False answer nums: {false}\n")
        print(f"Accuracy: {accuracy}\n")
        print(f"Cost: {cost}\n")

        result.append({"model_id": model_id, "correct": correct, "invalid": invalid, "false": false, "accuracy": accuracy, "cost": cost})
    return result

def plot_prompt_comparison(standard_accuracy: float, improved_accuracy: float,
                           out_path: str = "../../a4_tex/images/gsm8k_prompt_comparison.png") -> None:
    """
    Bar chart of model A's GSM8K accuracy under the standard prompt vs the
    improved prompt, for question 2bii.

    Args:
        standard_accuracy: accuracy of model A with standard_prompt_template.
        improved_accuracy: accuracy of model A with superior_prompt_template.
        out_path: where to write the figure.
    """
    labels = ["Standard prompt", "Improved prompt"]
    values = [standard_accuracy * 100, improved_accuracy * 100]
    colors = ["#2a78d6", "#eb6834"]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, values, width=0.42, color=colors, zorder=3)

    # Axis must start at 0 so the bar lengths stay proportional to accuracy.
    # Headroom above 100 keeps the value labels clear of the title.
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Accuracy on GSM8K (%)", color="#52514e")
    ax.set_title("Model A accuracy on GSM8K (n=100)", color="#0b0b0b",
                 fontsize=11, pad=14)

    # Recessive grid, horizontal only.
    ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)

    # Direct labels: only two bars, so label both.
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                ha="center", va="bottom", color="#0b0b0b", fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    result1 = eval_model_on_gsm8k()
    result2 = eval_model_on_gsm8k_with_improved_prompt()
    plot_prompt_comparison(result1[0]["accuracy"], result2[0]["accuracy"])
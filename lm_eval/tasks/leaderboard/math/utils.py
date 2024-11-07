import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger


try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )


INVALID_ANSWER = "[invalidanswer]"


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return (
        "Problem:\n"
        + doc["problem"]
        + "\nPlease provide your solution in the following format 'Final Answer: ###[final answer]'."
        + "\n\nSolution:\n"
    )



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": (
                "The expressions inside each square root must be non-negative. Therefore, $x - 2 \\geq 0$, so $x \\geq 2$, "
                "and $5 - x \\geq 0$, so $x \\leq 5$. Also, the denominator cannot be zero, so $5 - x > 0$, which gives $x < 5$. "
                "Therefore, the domain of the expression is $[2, 5)$.\n\n"
                "Final Answer: ###$[2, 5)$"
            ),
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": (
                "We have that $\\det(\\mathbf{A} \\mathbf{B}) = \\det(\\mathbf{A}) \\cdot \\det(\\mathbf{B}) = 2 \\times 12 = 24$.\n\n"
                "Final Answer: ###$24$"
            ),
            "few_shot": "1",
        },
        {
            "problem": (
                "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, "
                "how many times must Terrell lift them in order to lift the same total weight?"
            ),
            "solution": (
                "Terrell lifts two 20-pound weights 12 times, lifting a total weight of $2 \\times 20 \\times 12 = 480$ pounds. "
                "Using two 15-pound weights, each lift is $2 \\times 15 = 30$ pounds. Let $n$ be the number of lifts needed: "
                "$30n = 480$, so $n = 16$.\n\n"
                "Final Answer: ###$16$"
            ),
            "few_shot": "1",
        },
        {
            "problem": (
                "If the system of equations\n\n"
                "\\begin{align*}\n"
                "6x - 4y &= a, \\\\\n"
                "6y - 9x &= b.\n"
                "\\end{align*}\n"
                "has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\n"
                "find $\\frac{a}{b},$ assuming $b$ is nonzero."
            ),
            "solution": (
                "Multiply the first equation by $\\frac{3}{2}$: $9x - 6y = \\frac{3}{2}a$. "
                "Adding this to the second equation $6y - 9x = b$ gives $0 = \\frac{3}{2}a + b$, "
                "so $\\frac{a}{b} = -\\frac{2}{3}$.\n\n"
                "Final Answer: ###$-\\frac{2}{3}$"
            ),
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)

    if answer == INVALID_ANSWER:
        return {"exact_match": 0}

    if answer.strip() == doc["answer"].strip() or is_equiv(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    results = {
        "exact_match": retval,
    }
    return results


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    try:
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except AssertionError:
        return INVALID_ANSWER


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=1):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    # Ensure that the text ends with a newline to capture answers at the end of the string
    text += "\n"
    match = re.search(
        r"Final Answer : ###(.*?)(?:\n|$)",
        text
    )
    match_v1 = re.search(
        r"Final Answer :###(.*?)(?:\n|$)",
        text
    )
    match_v2 = re.search(
        r"Final Answer:###(.*?)(?:\n|$)",
        text
    )
    match_v3 = re.search(
        r"###(.*?)(?:\n|$)",
        text
    )
    match_v4 = re.search(
        r"Final Answer: (.*?)(?:\n|$)",
        text
    )
    if match or match_v1 or match_v2 or match_v3 or match_v4:
        try:
            return match.group(1).strip()
        except:
            try:
                return match_v1.group(1).strip()
            except:
                try:
                    return match_v2.group(1).strip()
                except:
                    try:
                        return match_v3.group(1).strip()
                    except:
                        try:
                            return match_v4.group(1).strip()
                        except:
                            return INVALID_ANSWER
    else:
        return INVALID_ANSWER

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer

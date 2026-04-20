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


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return (
        "Problème:\n"
        + doc["problem"]
        + "\n\n"
        + "Veuillez terminer votre solution en écrivant 'Réponse finale : ### [votre réponse]'.\n"
        + "Solution:\n"
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
            "problem": "Trouvez le domaine de l'expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
            "solution": "Les expressions sous chaque racine carrée doivent être non négatives. Donc, $x - 2 \\ge 0$, donc $x \\ge 2$, et $5 - x \\ge 0$, donc $x \\le 5$. De plus, le dénominateur ne peut pas être égal à zéro, donc $5 - x > 0$, ce qui donne $x < 5$. Par conséquent, le domaine de l'expression est $\\boxed{[2,5)}$.\nRéponse finale : ### $[2,5)$",
            "few_shot": "1",
        },
        {
            "problem": "Si $\\det \\mathbf{A} = 2$ et $\\det \\mathbf{B} = 12$, alors trouvez $\\det (\\mathbf{A} \\mathbf{B})$.",
            "solution": "Nous avons que $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}$.\nRéponse finale : ### $24$",
            "few_shot": "1",
        },
        {
            "problem": "Terrell soulève habituellement deux haltères de 20 livres 12 fois. S'il utilise deux haltères de 15 livres à la place, combien de fois Terrell doit-il les soulever pour obtenir le même poids total ?",
            "solution": "Si Terrell soulève deux haltères de 20 livres 12 fois, il soulève un total de $2\\cdot 12\\cdot20=480$ livres. S'il soulève à la place deux haltères de 15 livres $n$ fois, il soulèvera un total de $2\\cdot15\\cdot n=30n$ livres. En égalant cela à 480 livres, nous pouvons résoudre pour $n$ :\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nRéponse finale : ### $16$",
            "few_shot": "1",
        },
        {
            "problem": "Si le système d'équations\n\n\\begin{align*}\n6x - 4y &= a, \\\\\n6y - 9x &= b.\n\\end{align*}a une solution $(x, y)$ où $x$ et $y$ sont tous deux non nuls,\ntrouvez $\\frac{a}{b}$, en supposant que $b$ est non nul.",
            "solution": "Si nous multiplions la première équation par $-\\frac{3}{2}$, nous obtenons\n\n$$6y - 9x = -\\frac{3}{2}a.$$Comme nous savons aussi que $6y - 9x = b$, nous avons\n\n$$-\\frac{3}{2}a = b \\Rightarrow \\frac{a}{b} = \\boxed{-\\frac{2}{3}}.$$\nRéponse finale : ### $-\\frac{2}{3}$",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)    
    if is_equiv(answer, doc["answer"]):
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
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


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
        with timeout(seconds=5):
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
        r"Réponse finale : ###(.*?)(?:\n|$)",
        text
    )
    match_v1 = re.search(
        r"Réponse finale :###(.*?)(?:\n|$)",
        text
    )
    match_v2 = re.search(
        r"Réponse finale:###(.*?)(?:\n|$)",
        text
    )
    match_v3 = re.search(
        r"###(.*?)(?:\n|$)",
        text
    )
    match_v4 = re.search(
        r"Réponse finale: (.*?)(?:\n|$)",
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
    "carré",
    "manières",
    "entiers",
    "dollars",
    "mph",
    "pouces",
    "pieds",
    "heures",
    "km",
    "unités",
    "\\ldots",
    "sue",
    "points",
    "pieds",
    "minutes",
    "chiffres",
    "cents",
    "degrés",
    "cm",
    "gm",
    "livres",
    "mètres",
    "repas",
    "arêtes",
    "étudiants",
    "billetsdenfants",
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

"""CABIN (Comprehensive Assessment of Basic Interests) benchmark.

Evaluates LLMs on 164 career interest items across 41 categories (4 items
each) using a 1-5 Likert scale. No correct answer — measures vocational
preferences. Similarity is computed from mean absolute difference of
category-level averages.

Reference: Su, Tay, Liao, Zhang & Rounds (2019)
Source:    https://github.com/CUHK-ARISE/PsychoBench (questionnaires.json)
"""

import re
from typing import Any, override

from benchmarks.base import Benchmark, BenchmarkResult
from src.utils.concurrency import run_tasks

# ---------------------------------------------------------------------------
# All 164 CABIN items (from PsychoBench questionnaires.json)
# ---------------------------------------------------------------------------
CABIN_QUESTIONS: dict[int, str] = {
    1: "Repair car engines.",
    2: "Perform aircraft maintenance.",
    3: "Maintain wind turbine generators.",
    4: "Install radio communication systems.",
    5: "Build wood wall shelves.",
    6: "Build kitchen cabinets.",
    7: "Sand and refinish a piece of furniture.",
    8: "Build a fence.",
    9: "Drive a bus.",
    10: "Drive a delivery truck.",
    11: "Operate a train.",
    12: "Operate a crane to move freight and cargo.",
    13: "Load and unload aircraft baggage.",
    14: "Load and unload cargo.",
    15: "Move building materials on construction sites.",
    16: "Pack and move products in a warehouse.",
    17: "Arrest suspects of criminal acts.",
    18: "Conduct surveillance of suspects.",
    19: "Inspect people and vehicles for illegal goods.",
    20: "Investigate reports of organized crime.",
    21: "Farm and harvest crops.",
    22: "Inspect orchards to detect diseases or pests.",
    23: "Learn about soil and climate requirements of various plants.",
    24: "Apply principles of soil science to conserve land.",
    25: "Water and fertilize garden plants.",
    26: "Survey forest areas and access roads.",
    27: "Plant trees in a nature preserve.",
    28: "Work to restore a wildlife habitat.",
    29: "Treat and care for injured animals.",
    30: "Feed and bathe animals in a zoo.",
    31: "Exercise animals daily to keep them healthy.",
    32: "Find stray animals and take them to a shelter.",
    33: "Play a team or individual sport.",
    34: "Participate in athletic events.",
    35: "Train for a competitive sport.",
    36: "Coach practice sessions for a sports team.",
    37: "Design a structure that can withstand heavy wind.",
    38: "Develop lighter and stronger materials for new products.",
    39: "Redesign a production line to improve its efficiency.",
    40: "Improve the human-machine interface of an operation system.",
    41: "Study the formation and evolution of galaxies.",
    42: "Analyze a mineral sample found on Mars.",
    43: "Investigate the molecular structure of an unknown substance.",
    44: "Study the causes for earthquakes and tsunamis.",
    45: "Map human gene structure.",
    46: "Study the physiological structure of animals.",
    47: "Investigate the genetic sequence of organisms.",
    48: "Research newly discovered bacteria with laboratory experiments.",
    49: "Examine how viruses infect the human body.",
    50: "Investigate the cause of a chronic health problem.",
    51: "Research the side effects of a medicine.",
    52: "Investigate prevention methods for diseases.",
    53: "Study cultural differences between groups.",
    54: "Investigate how poverty influences educational attainment.",
    55: "Study the effects of public policy on violence reduction.",
    56: "Research why people have stereotypes and prejudice.",
    57: "Study the history of an ancient society.",
    58: "Study various branches of philosophy.",
    59: "Compare the modern history of different countries.",
    60: "Document the traditions of a remote community.",
    61: "Solve mathematical problems.",
    62: "Learn about a new theory in geometry.",
    63: "Use mathematical equations to solve practical problems.",
    64: "Develop a statistical model to explain a phenomenon.",
    65: "Test and compare different software.",
    66: "Create a new computer database.",
    67: "Monitor the daily performance of computer systems.",
    68: "Diagnose and resolve computer hardware or software problems.",
    69: "Sketch a picture.",
    70: "Paint a landscape.",
    71: "Draw illustrations for a book.",
    72: "Create a unique piece of artwork.",
    73: "Create a piece of artistic and functional furniture.",
    74: "Create the set for a movie or stage play.",
    75: "Design the layout and lighting of an exhibition.",
    76: "Design unique packaging for a product.",
    77: "Perform on stage for a group of people.",
    78: "Act in a play.",
    79: "Act out an emotional movie scene.",
    80: "Perform comedy to entertain an audience.",
    81: "Play a musical instrument.",
    82: "Compose an original piece of music.",
    83: "Play in a band.",
    84: "Arrange background music for a show.",
    85: "Write a novel.",
    86: "Write short stories.",
    87: "Compose a poem.",
    88: "Study creative writing.",
    89: "Direct a TV show.",
    90: "Write a movie screenplay.",
    91: "Host a radio program.",
    92: "Develop a podcast series.",
    93: "Select ingredients to prepare food.",
    94: "Create the recipe for a new dish.",
    95: "Create a new cooking technique to enhance flavor.",
    96: "Learn about required temperature and time for baking pastries.",
    97: "Teach students a new set of skills.",
    98: "Explain a topic to someone with no prior knowledge of the subject.",
    99: "Teach a beginner how to perform a task.",
    100: "Teach visitors on educational field trips.",
    101: "Volunteer at a community service center.",
    102: "Help someone overcome an obstacle in personal life.",
    103: "Provide aid to students from underprivileged backgrounds.",
    104: "Assist people with disabilities in finding employment.",
    105: "Treat patients for acute illnesses or injuries.",
    106: "Care for patients in critical condition.",
    107: "Monitor patient reactions to medicines.",
    108: "Formulate treatment plans for patients.",
    109: "Provide spiritual guidance for others.",
    110: "Explain a religious text to people.",
    111: "Teach religious beliefs and rituals.",
    112: "Work with a religious youth group.",
    113: "Arrange travel plans and accommodations for clients.",
    114: "Greet guests and answer questions at an information desk.",
    115: "Help clients plan for their special occasions.",
    116: "Organize recreational activities for clients.",
    117: "Coach others to develop leadership skills.",
    118: "Coach people to prepare for job interviews.",
    119: "Advise people in meeting their professional goals.",
    120: "Instruct clients in effective communication techniques.",
    121: "Negotiate a business deal.",
    122: "Set up a string of small business enterprises.",
    123: "Expand a business to incorporate a new line of products.",
    124: "Beat competitors through strategic business practices.",
    125: "Persuade customers to try a new product.",
    126: "Increase sales for a company during a promotion week.",
    127: "Sell services to a target group of people.",
    128: "Learn tactics to be effective at sales.",
    129: "Lead an advertising campaign.",
    130: "Market a company on social media platforms.",
    131: "Coordinate marketing activities to promote a new product.",
    132: "Distribute promotional materials to advertise an event.",
    133: "Make investment decisions based on financial data.",
    134: "Analyze the financial information of a company.",
    135: "Project future expenditures of a business.",
    136: "Assess potential risks and gains of an investment.",
    137: "Prepare employee payroll.",
    138: "Monitor account balance and prepare monthly statements.",
    139: "Keep accounting records for a company.",
    140: "Calculate tax deductions for a business.",
    141: "Hire employees and process hiring-related paperwork.",
    142: "Conduct orientation sessions for new workers.",
    143: "Explain company policies and benefits to employees.",
    144: "Conduct surveys of employee satisfaction.",
    145: "Enter personnel records into a computer program.",
    146: "Catalog files in an office.",
    147: "Print and disseminate documents to be used at a conference.",
    148: "Keep track of customer requests.",
    149: "Manage a medium-sized organization.",
    150: "Supervise a large number of workers.",
    151: "Serve as the chairperson for a corporate board.",
    152: "Serve as the president of a professional association.",
    153: "Present your ideas at a conference.",
    154: "Speak as the representative of an organization.",
    155: "Be the speaker at a fund-raising event for a worthy cause.",
    156: "Make a public speech to raise awareness of community issues.",
    157: "Run for a political office.",
    158: "Be the head of the city council.",
    159: "Lead a committee to make policy decisions.",
    160: "Assume political leadership responsibilities.",
    161: "Defend a client against a legal charge.",
    162: "Present logical arguments in a courtroom.",
    163: "Provide compelling evidence for a trial.",
    164: "Resolve legal disputes between parties.",
}

# 41 categories: (name, question_ids, crowd_mean, crowd_std)
CABIN_CATEGORIES: list[tuple[str, list[int], float, float]] = [
    ("Mechanics/Electronics", [1, 2, 3, 4], 2.41, 1.29),
    ("Construction/WoodWork", [5, 6, 7, 8], 3.10, 1.29),
    ("Transportation/Machine Operation", [9, 10, 11, 12], 2.49, 1.23),
    ("Physical/Manual Labor", [13, 14, 15, 16], 2.24, 1.19),
    ("Protective Service", [17, 18, 19, 20], 3.02, 1.35),
    ("Agriculture", [21, 22, 23, 24], 3.01, 1.24),
    ("Nature/Outdoors", [25, 26, 27, 28], 3.59, 1.13),
    ("Animal Service", [29, 30, 31, 32], 3.65, 1.20),
    ("Athletics", [33, 34, 35, 36], 3.26, 1.28),
    ("Engineering", [37, 38, 39, 40], 2.93, 1.29),
    ("Physical Science", [41, 42, 43, 44], 3.25, 1.31),
    ("Life Science", [45, 46, 47, 48], 3.03, 1.23),
    ("Medical Science", [49, 50, 51, 52], 3.31, 1.30),
    ("Social Science", [53, 54, 55, 56], 3.37, 1.16),
    ("Humanities", [57, 58, 59, 60], 3.33, 1.20),
    ("Mathematics/Statistics", [61, 62, 63, 64], 2.93, 1.35),
    ("Information Technology", [65, 66, 67, 68], 2.93, 1.29),
    ("Visual Arts", [69, 70, 71, 72], 3.32, 1.26),
    ("Applied Arts and Design", [73, 74, 75, 76], 3.16, 1.22),
    ("Performing Arts", [77, 78, 79, 80], 2.75, 1.38),
    ("Music", [81, 82, 83, 84], 3.16, 1.29),
    ("Writing", [85, 86, 87, 88], 3.19, 1.27),
    ("Media", [89, 90, 91, 92], 3.01, 1.24),
    ("Culinary Art", [93, 94, 95, 96], 3.82, 1.08),
    ("Teaching/Education", [97, 98, 99, 100], 3.74, 1.07),
    ("Social Service", [101, 102, 103, 104], 3.87, 0.96),
    ("Health Care Service", [105, 106, 107, 108], 2.88, 1.33),
    ("Religious Activities", [109, 110, 111, 112], 2.65, 1.35),
    ("Personal Service", [113, 114, 115, 116], 3.32, 1.20),
    ("Professional Advising", [117, 118, 119, 120], 3.30, 1.24),
    ("Business Initiatives", [121, 122, 123, 124], 3.17, 1.25),
    ("Sales", [125, 126, 127, 128], 3.07, 1.25),
    ("Marketing/Advertising", [129, 130, 131, 132], 2.87, 1.23),
    ("Finance", [133, 134, 135, 136], 3.07, 1.32),
    ("Accounting", [137, 138, 139, 140], 3.00, 1.27),
    ("Human Resources", [141, 142, 143, 144], 3.27, 1.20),
    ("Office Work", [145, 146, 147, 148], 3.27, 1.14),
    ("Management/Administration", [149, 150, 151, 152], 3.05, 1.27),
    ("Public Speaking", [153, 154, 155, 156], 2.86, 1.38),
    ("Politics", [157, 158, 159, 160], 2.32, 1.33),
    ("Law", [161, 162, 163, 164], 3.08, 1.30),
]

# Build question-to-category lookup
_QID_TO_CATEGORY: dict[int, str] = {}
for _cat_name, _qids, _, _ in CABIN_CATEGORIES:
    for _qid in _qids:
        _QID_TO_CATEGORY[_qid] = _cat_name

LETTER_TO_SCORE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

CABIN_PROMPT = """\
Rate the following activity based on how much you would like to do it:

"{activity}"

A) Dislike
B) Slightly Dislike
C) Neither Like nor Dislike
D) Slightly Like
E) Like Very Much

Your response should be in the following format:
Explanation: {{your explanation for your choice}}
Answer: {{the letter of your chosen answer}}"""


class CABIN(Benchmark):
    """CABIN (Comprehensive Assessment of Basic Interests) benchmark.

    164 career interest items across 41 categories. Each item is rated 1-5
    on a Likert scale. No correct answer — measures vocational preferences.
    Similarity is computed from mean absolute difference of category averages.
    """

    name = "cabin"

    def __init__(
        self,
        max_items: int | None = None,
        max_workers: int | None = None,
        store_traces: bool = False,
    ) -> None:
        """
        Args:
            max_items: Limit number of items (for cheaper runs).
            max_workers: Max concurrent LLM calls. None = one per question.
            store_traces: If True, store full LLM response in raw_responses.
        """
        items = [
            {"id": qid, "text": text}
            for qid, text in CABIN_QUESTIONS.items()
        ]

        if max_items:
            from benchmarks.sampling import stratified_sample
            items = stratified_sample(
                items,
                lambda q: _QID_TO_CATEGORY.get(q["id"], "unknown"),
                max_items,
            )

        self.questions = items
        self.max_workers = max_workers
        self.store_traces = store_traces

    @staticmethod
    def _parse_answer(response: str) -> int | None:
        """Extract the chosen rating (1-5) from the response."""
        match = re.search(r"Answer:\s*([A-Ea-e])", response)
        if match:
            return LETTER_TO_SCORE.get(match.group(1).upper())

        # Fallback: last standalone A-E letter
        letters = re.findall(r"\b([A-E])\b", response)
        for letter in reversed(letters):
            return LETTER_TO_SCORE.get(letter)

        return None

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        """Run CABIN benchmark on an agent."""
        def _call(item: dict) -> tuple[str, str, int | None]:
            prompt = CABIN_PROMPT.format(activity=item["text"])
            response, trace_id = agent.invoke(prompt)
            rating = self._parse_answer(response)
            return response, trace_id, rating

        call_results = run_tasks(
            self.questions,
            _call,
            max_workers=self.max_workers,
            desc=f"cabin [{agent.name.split('/')[-1][:25]}]",
        )

        raw_responses: list[dict] = []
        answer_vector: dict[str, int | None] = {}
        category_sums: dict[str, float] = {}
        category_counts: dict[str, int] = {}

        for item, (response, trace_id, rating) in zip(self.questions, call_results):
            qid = item["id"]
            answer_vector[str(qid)] = rating

            cat = _QID_TO_CATEGORY.get(qid)
            if rating is not None and cat is not None:
                category_sums[cat] = category_sums.get(cat, 0.0) + rating
                category_counts[cat] = category_counts.get(cat, 0) + 1

            entry: dict[str, Any] = {
                "qid": qid,
                "question_text": item["text"],
                "category": cat,
                "rating": rating,
                "trace_id": trace_id,
            }
            if self.store_traces:
                entry["full_response"] = response

            raw_responses.append(entry)

        category_scores = {
            cat: category_sums[cat] / category_counts[cat]
            for cat in category_sums
            if category_counts.get(cat, 0) > 0
        }

        scores: dict[str, Any] = {
            "answer_vector": answer_vector,
            "category_scores": category_scores,
        }

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores=scores,
            raw_responses=raw_responses,
        )

    @override
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Quadratic Weighted Kappa linearly rescaled to [0, 1] (returned ×100).

        score = (1 − ½ · D_obs / D_exp) * 100

        D_obs = (1/N) Σ_i (a_i − b_i)²       over paired ratings
        D_exp = (1/N²) Σ_i Σ_j (a_i − b_j)²  cross-product over empirical marginals

        Equivalent to (k_w + 1)/2 where k_w = 1 − D_obs/D_exp is standard QWK.
        Bounded in [0, 1] without clipping because k_w ∈ [−1, +1] always
        (D_obs − D_exp = −2·cov, |cov| ≤ σ_a σ_b, D_exp ≥ 2 σ_a σ_b).

        100 = identical, 50 ≈ chance under marginals, 0 = max anti-correlation.
        """
        vec_a = result_a.scores.get("answer_vector", {})
        vec_b = result_b.scores.get("answer_vector", {})

        common = set(vec_a.keys()) & set(vec_b.keys())
        valid = [(vec_a[qid], vec_b[qid]) for qid in common
                 if vec_a[qid] is not None and vec_b[qid] is not None]

        if not valid:
            return 50.0

        xs = [a for a, _ in valid]
        ys = [b for _, b in valid]
        n = len(valid)

        d_obs = sum((a - b) ** 2 for a, b in valid) / n
        d_exp = sum((a - b) ** 2 for a in xs for b in ys) / (n * n)

        if d_exp == 0:
            return 100.0  

        return (1 - 0.5 * d_obs / d_exp) * 100

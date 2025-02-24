from dataclasses import dataclass
import sys
import gamspy as gp
import numpy as np


coords = tuple[int, int]


@dataclass
class ProblemInput:
    num_blocks: int
    block_lens: list[int]
    min_len: int
    max_len: int
    letters: set[str]
    matches: list[tuple[int, int, int, int]]
    blocks: list[tuple[coords, coords]]


def main(problem: ProblemInput):
    min_len = problem.min_len
    max_len = problem.max_len
    letters = problem.letters
    num_blocks = problem.num_blocks
    matches = [(str(mm) for mm in m) for m in problem.matches]

    vocabulary = []

    block_len_set = set(problem.block_lens)
    with open("filtered_words.txt", "r") as f:
        for line in f:
            line = line.strip()

            if len(line) == 1:
                continue

            if len(line) > max_len or len(line) < min_len:
                continue

            if line.endswith("'s"):
                continue

            if len(line) not in block_len_set:
                continue

            if set(line) <= letters:
                vocabulary.append(line)

    if len(vocabulary) == 0:
        raise Exception(
            "Letters you picked combined with the blocks does not match any words in the dictionary"
        )

    letters = list(letters)

    m = gp.Container()

    i = gp.Set(
        m, "i", records=range(1, max_len + 1), description="set for letter places"
    )

    j = gp.Set(
        m,
        name="j",
        records=range(1, num_blocks + 1),
        description="Number of blocks in the puzzle",
    )

    ii = gp.Alias(m, "ii", alias_with=i)
    jj = gp.Alias(m, "jj", alias_with=j)

    if len(matches) == 0:
        matches = None

    common = gp.Set(
        m,
        name="common",
        domain=[j, j, i, i],
        description="Set indicating which char should be common",
        records=matches,
    )

    k = gp.Set(
        m, name="k", records=vocabulary, description="Words that we can choose from"
    )

    kk = gp.Alias(m, name="kk", alias_with=k)

    l = gp.Set(m, name="l", records=letters, description="Letters that are allowed")

    LB = gp.Parameter(
        m,
        domain=[j],
        description="Length of the block j",
        records=np.array(problem.block_lens),
    )

    wlens = np.array([len(x) for x in vocabulary])

    LW = gp.Parameter(
        m, name="LW", domain=[k], description="Length of words", records=wlens
    )

    wlet = gp.Parameter(m, name="wlet", domain=[k, i, l])

    print(len(vocabulary))
    for word in vocabulary:
        for ind, cha in enumerate(word):
            wlet[word, str(ind + 1), cha] = 1

    w = gp.Variable(
        m,
        name="w",
        domain=[j, k],
        type="binary",
        description="which word is assigned to which place",
    )

    # allow only correct placing
    w.fx[j, k].where[LW[k] != LB[j]] = 0

    # assign words to every position
    eq1 = gp.Equation(m, domain=[j])
    eq1[j] = gp.Sum(k, w[j, k]) == 1

    # use a word atmost once
    eq2 = gp.Equation(m, domain=[k])
    eq2[k] = gp.Sum(j, w[j, k]) <= 1

    big_M = 3

    # use common letters
    eq3 = gp.Equation(m, domain=[j, jj, i, ii, l])
    eq3[j, jj, i, ii, l].where[
        common[j, jj, i, ii]
    ] = (
        gp.Sum(k.where[wlet[k, i, l]], w[j, k]) ==
        gp.Sum(k.where[wlet[k, ii, l]], w[jj, k])
    )

    model = gp.Model(
        m,
        equations=m.getEquations(),
        problem="mip",
    )

    result = model.solve(output=sys.stdout)
    if "IntegerInfeasible" in str(result):
        raise Exception("Problem is infeasible, no solution found!")

    data = w.records
    data = data[data["level"] == 1]

    summary = []
    for index, row in data.iterrows():
        summary.append((row["j"], row["k"]))

    return result, summary

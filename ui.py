from functools import partial

import gamspy as gp
import numpy as np
import streamlit as st

from problem import main, ProblemInput


coords = tuple[int, int]


def get_block_len(block: tuple[coords, coords]):
    h_diff = block[1][0] - block[0][0]
    w_diff = block[1][1] - block[0][1]
    return max(w_diff, h_diff) + 1


def get_block_letter(block: tuple[coords, coords], coord: coords):
    h_diff = coord[0] - block[0][0]
    w_diff = coord[1] - block[0][1]
    return max(w_diff, h_diff) + 1


def get_block_horizontal(values: np.ndarray):
    h, w = values.shape
    h_blocks = []
    for ih in range(h):
        state = None
        for iw in range(w):
            observation = values[ih, iw]
            if observation:
                if state is None:
                    state = (ih, iw)
                else:
                    continue
            else:
                if state is None:
                    continue
                else:
                    if iw != state[1] + 1:  # don't allow blocks with size 1
                        h_blocks.append((state, (ih, iw - 1)))

                    state = None

        if state is not None and iw != state[1] + 1:
            h_blocks.append((state, (ih, iw)))

    return h_blocks


def get_block_vertical(values: np.ndarray):
    h, w = values.shape
    v_blocks = []
    for iw in range(w):
        state = None
        for ih in range(h):
            observation = values[ih, iw]
            if observation:
                if state is None:
                    state = (ih, iw)
                else:
                    continue
            else:
                if state is None:
                    continue
                else:
                    if ih != state[0] + 1:  # don't allow blocks with size 1
                        v_blocks.append((state, (ih - 1, iw)))

                    state = None

        if state is not None and ih != state[0] + 1:
            v_blocks.append((state, (ih, iw)))

    return v_blocks


def detect_structure(values: np.ndarray) -> ProblemInput:
    # blocks are numbered first horizontal blocks then vertical ones
    h_blocks = get_block_horizontal(values)
    v_blocks = get_block_vertical(values)
    blocks = [*h_blocks, *v_blocks]

    if len(blocks) == 0:
        e = Exception("Please select some blocks, and blocks with size 1 do not count")
        st.exception(e)
        raise e

    num_blocks = len(blocks)

    block_lens = [get_block_len(x) for x in blocks]
    min_len = min(block_lens)
    max_len = max(block_lens)

    matches = []
    block_map = {}
    for i, block in enumerate(blocks):
        for ih in range(block[0][0], block[1][0] + 1):
            for iw in range(block[0][1], block[1][1] + 1):
                if (ih, iw) not in block_map:
                    block_map[(ih, iw)] = i  # 1-indexed
                else:
                    first = block_map[(ih, iw)]
                    second = i
                    first_letter = get_block_letter(blocks[first], (ih, iw))
                    second_letter = get_block_letter(block, (ih, iw))
                    matches.append((first + 1, second + 1, first_letter, second_letter))

    print(matches)
    print("Number of block:", num_blocks)
    print("Block lens:", block_lens)
    print("Min len:", min_len)
    print("Max len:", max_len)
    print(h_blocks)
    print(v_blocks)

    return ProblemInput(
        num_blocks=num_blocks,
        block_lens=block_lens,
        min_len=min_len,
        max_len=max_len,
        letters={"c", "a", "m", "p", "u", "l"},
        matches=matches,
        blocks=blocks,
    )

row1 = "abcdefgh"
row2 = "ijklmnop"
row3 = "qrstuvwxyz"

letters_selected = {}

st.header("Letters")

cols1 = st.columns(10)
cols2 = st.columns(10)
cols3 = st.columns(10)

for i, r in enumerate(row1):
    with cols1[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

for i, r in enumerate(row2):
    with cols2[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

for i, r in enumerate(row3):
    with cols3[i]:
        letters_selected[r] = st.checkbox(r, key=f"{r}")

letters_selected = {k for k in letters_selected if letters_selected[k]}

st.write("\n")


row_count = 10
col_count = 10

def toggle_state(row, i):
    st.session_state[(row, i)] = not st.session_state[(row, i)]

st.header("Problem Structure")

for row in range(row_count):
    cols = st.columns(10)
    for i, col in enumerate(cols):
        if (row, i) not in st.session_state:
            st.session_state[(row, i)] = False

        with col:
            color = "primary" if st.session_state[(row, i)] else "secondary"
            st.button(
                key=f"{row}x{i}",
                label=" ",
                on_click=partial(toggle_state, row, i),
                type=color,
                use_container_width=True,
            )


st.header("Solution")


if "solution" not in st.session_state:
    st.session_state["solution"] = {}

if st.button("Solve"):
    values = []
    for row in range(row_count):
        temp = []
        for col in range(col_count):
            temp.append(st.session_state[(row, col)])

        values.append(temp)

    values = np.array(values)

    if len(letters_selected) == 0:
        e = Exception("You did not select any letters!")
        st.exception(e)
        raise e

    problem = detect_structure(values)
    problem.letters = letters_selected

    try:
        result, summary = main(problem)
    except gp.exceptions.GamspyException as e:
        if "license error" in str(e):
            st.write("License Error!")
        else:
            st.exception(e)
            raise e
    except Exception as e:
        st.exception(e)
        raise e
    else:
        new_solution = {}
        for index, word in summary:
            index = int(index)
            block = problem.blocks[index - 1]
            letter_index = 0
            for ih in range(block[0][0], block[1][0] + 1):
                for iw in range(block[0][1], block[1][1] + 1):
                    l = word[letter_index]
                    letter_index += 1
                    new_solution[(ih, iw)] = l

        st.session_state["solution"] = new_solution

        st.write(result)

        st.markdown("""
            <style>.element-container:has(#button-after) + div button {
                color: black;
                background-color: rgb(255, 75, 75);
                    }
                    #button-after {
                        display: none;
                    }
                    #disabled-after {
                        display: none;
                    }
                    .element-container:has(#button-after) {
                    display: none;
                    }
                    .element-container:has(#disabled-after) {
                    display: none;
                    }
                    </style>""", unsafe_allow_html=True)

        for row in range(row_count):
            cols = st.columns(10)
            for i, col in enumerate(cols):
                with col:
                    color = "primary" if st.session_state[(row, i)] else "secondary"
                    label = " "
                    if (row, i) in st.session_state["solution"]:
                        label = st.session_state["solution"][(row, i)]
                        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span id="disabled-after"></span>', unsafe_allow_html=True)

                    st.button(
                        key=f"sol_{row}x{i}",
                        label=label,
                        type=color,
                        use_container_width=True,
                        disabled=True,
                    )

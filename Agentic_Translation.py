import streamlit as st
import streamlit.components.v1 as components
from collections import defaultdict
import matplotlib.pyplot as plt
import difflib
import numpy as np

import json
import re

st.set_page_config(page_title="ISA-N", layout='wide')

source_lang, tgt_lang = "aarch64", "aarch64+sve"
stages = ["GENERATION", "USE_INSN", "ASSEMBLY", "QEMU", "WRONG_OUTPUT", "SUCCESS"]
wanted_regs = ["z", "Z", "p", "P"]
error_pattern = re.compile(r'(?i)error:\s*(.*?)(?=\n)', re.MULTILINE)
possible_error_prefixes = {
    "unrecognized instruction mnemonic",
    "unexpected token",
    "expected identifier",
    "unknown directive",
    "unknown token in expression",
    "production requires",
    "instruction requires",
    "index must be a",
    "immediate must be a",
    "instruction is unpredictable when",
}
def check_used_insns(generated_code, wanted_insns, wanted_registers):
    if not generated_code: return False
    if any([re.search(r'\n\s*' + insn + r'\s', generated_code) is not None for insn in wanted_insns]):
        return True
    if wanted_registers and any([re.search(r'[^\d\w]' + reg + r'\d+[^\d\w]', generated_code.lower()) for reg in wanted_registers]):
        return True
    return False


# LOAD IN DATA
orig_files_data = {}
input_dataset_path = f"translate_with_manual/translate_{tgt_lang}_dataset.json"
for ex in json.load(open(input_dataset_path)):
    filename = re.search(r'```[\s\S]+.file\s*\"(.+\.c)\"\n[\s\S]+```', ex["prompt"])
    if filename is None: breakpoint()
    orig_code = re.search(r'```aarch64-unknown-linux-gnu\n([^`]+)```', ex["prompt"]).group(1)
    filename = filename.group(1)
    orig_files_data[filename] = {"true_exc_output": ex["true_exc_output"], "original_code": orig_code, "gold": ex["completion"]}

filenames = [
	"translate_with_manual/DeepSeek-R1-Distill-Qwen-32B_2_28.json",
	"translate_with_manual/DeepSeek-R1-Distill-Qwen-32B_3_03.json",
]
filename = st.selectbox("Which examples would you like to see?", filenames)
experiment_results = json.load(open(filename))

examples = {ex["orig_filename"]: ex for ex in experiment_results}
successes = set()

def render_whole_experiment_summary(generations):
    # shows role of feedback
    ex_to_attempts = defaultdict(lambda: defaultdict(str))
    assembly_errors_by_iteration = defaultdict(lambda: defaultdict(int))
    
    # shows success rate: example -> best it does
    # final_stage = defaultdict(lambda: defaultdict(str))

    max_num_iterations = 0
    for ex in generations:
        wanted_insns = list(ex["chunks"].keys())
        filename = ex["orig_filename"]
        if filename not in ex_to_attempts: ex_to_attempts[filename] = []
        for attempts in ex["samples"]:
            ex_to_attempts[filename].append([])
            for iteration, sample in enumerate(attempts):
                max_num_iterations = max(iteration, max_num_iterations)
                tested_code = sample["tested_code"]
                results = sample["results"]
                if tested_code is None:
                    this_error_stage_index = stages.index("GENERATION")
                elif not check_used_insns(tested_code, wanted_insns, wanted_regs):
                    this_error_stage_index = stages.index("USE_INSN")
                elif results[0] is None: 
                    # check whether the outputs match.
                    if orig_files_data[filename]["true_exc_output"] == results[1]:
                        this_error_stage_index = stages.index("SUCCESS")
                        successes.add(filename)
                    else: 
                        this_error_stage_index = stages.index("WRONG_OUTPUT")
                elif results[0].startswith("ASSEMBLE AND L"):
                    this_error_stage_index = stages.index("ASSEMBLY")
                    matches = error_pattern.findall(results[1])
                    for match in matches:
                        error_message = match.strip().lower()
                        error_category = error_message
                        for poss_prefix in possible_error_prefixes:
                            if error_message.startswith(poss_prefix): 
                                error_category = poss_prefix
                                break
                        assembly_errors_by_iteration[iteration][error_category] += 1
                elif results[0] == "QEMU":
                    this_error_stage_index = stages.index("QEMU")
                else: breakpoint()
            

                ex_to_attempts[filename][-1].append(stages[this_error_stage_index])
                # if filename in final_stage:
                #     final_stage[filename] = max(this_error_stage_index, final_stage[filename])
                # else:
                #     final_stage[filename] = this_error_stage_index

    stages_col, asm_col = st.columns(2)
    # stages_col, pie_col, asm_col = st.columns(3)
    iterations = list(range(max_num_iterations+1))

    with stages_col:
        # Stacked bar plot of errors by stage per iteration
        it_to_ratios = [{} for _ in iterations]
        for filename, attempts in ex_to_attempts.items():
            for it in iterations:
                if filename not in it_to_ratios[it]: it_to_ratios[it][filename] = []
                for attempt_chain in attempts:
                    if it >= len(attempt_chain):
                        it_to_ratios[it][filename].append(attempt_chain[-1])
                    else:
                        it_to_ratios[it][filename].append(attempt_chain[it])
        stage_counts_per_iteration = {stage: [] for stage in stages}

        for it, example_info in enumerate(it_to_ratios):
            for stage in stages:
                stage_to_ratios = [ex_it_stages_list.count(stage) / len(ex_it_stages_list) for ex_it_stages_list in example_info.values()]   
                stage_counts_per_iteration[stage].append(sum(stage_to_ratios))

        # x axis: iterations
        # stacked bar: each segment of stack is stage
        x = np.arange(len(iterations))
        width = 0.6
        bottom_vals = np.zeros(len(iterations))

        plt.figure(figsize=(10, 6))

        for stage in stages:
            plt.bar(x, stage_counts_per_iteration[stage], width, label=stage, bottom=bottom_vals)
            bottom_vals += np.array(stage_counts_per_iteration[stage])
            
        plt.xlabel("Feedback Iteration")
        plt.ylabel("Examples Status Breakdown")
        plt.ylim(bottom=0)
        plt.xticks(x, iterations)
        plt.title(f"Errors by Stage per Feedback Iteration ({len(it_to_ratios[0])} examples)")
        plt.legend()
        plt.grid()
        st.pyplot(plt.gcf(), )
        plt.close()
    
    # with pie_col:
    #     # pie chart of final_stage, where each slice is number examples in stage
    #     stage_counts = {stage: 0 for stage in stages}

    #     for stage_index in final_stage.values():
    #         stage_counts[stages[stage_index]] += 1

    #     plt.figure(figsize=(8, 8))
    #     plt.pie(stage_counts.values(), labels=stage_counts.keys(), autopct='%1.1f%%', startangle=140)
    #     plt.title(f"Final Outcome Distribution ({len(it_to_ratios[0])} total)")
    #     st.pyplot(plt.gcf(), )
    #     plt.close()
    
    with asm_col:
        # Plot breakdown of assembly errors by feedback iteration
        plt.figure(figsize=(12, 6))
        # Filter for top error categories
        error_totals = defaultdict(int)
        for iteration_errors in assembly_errors_by_iteration.values():
            for error, count in iteration_errors.items():
                error_totals[error] += count
        
        sorted_errors = sorted(error_totals.items(), key=lambda x: x[1], reverse=True)
        top_errors = {error for error, _ in sorted_errors[:9]}
        for iteration, errors in assembly_errors_by_iteration.items():
            other_count = sum(count for error, count in errors.items() if error not in top_errors)
            assembly_errors_by_iteration[iteration] = {error: count for error, count in errors.items() if error in top_errors}
            assembly_errors_by_iteration[iteration]["Other"] = other_count
        
        # Sort legend by most frequent error
        legend_order = [error for (error, _) in sorted_errors[:9]] + ["Other"]
        
        # all_assembly_errors = list(set(err for round_errors in assembly_errors_by_iteration.values() for err in round_errors))
        iterations = sorted(assembly_errors_by_iteration.keys())
        
        for error_type in legend_order:
            error_counts = [assembly_errors_by_iteration[iteration].get(error_type, 0) for iteration in iterations]
            plt.plot(iterations, error_counts, marker='o', linestyle='-', label=error_type)
        
        plt.xlabel("Feedback Iteration")
        plt.ylabel("Assembly Error Count")
        plt.ylim(bottom=0)
        plt.title(f"Breakdown of Assembly Errors by Feedback Iteration")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.grid()
        st.pyplot(plt.gcf(), )
        plt.close()

render_whole_experiment_summary(experiment_results)
st.subheader("Successes:"+ str(successes))

example_key = st.selectbox("Choose your example", list(examples.keys()))
example = examples[example_key]

def render_for_example(example):
    filename = example["orig_filename"]
    wanted_insns = list(example["chunks"].keys())

    chunk_expanders = []
    for chunk_name, chunk_description in example["chunks"].items():
        chunk_expanders.append(st.expander(chunk_name))
        chunk_expanders[-1].write(chunk_description)

    attempt_tabs = []
    for sample_idx, sample in enumerate(example["samples"]):
        eventually_succeeds = False
        for it_data in sample:
            tested_code = it_data['tested_code']
            results = it_data['results']
            if tested_code and (results[0] is None) and orig_files_data[filename]["true_exc_output"] == results[1]:
                if check_used_insns(tested_code, wanted_insns, wanted_regs):
                    eventually_succeeds = True

        attempt_tabs.append(f"Attempt {sample_idx} ({'succeeds' if eventually_succeeds else 'fails'})")

    tabs = st.tabs(attempt_tabs)
    for sample_idx, tab in enumerate(tabs):
        with tab:
            assembly_errors_by_iteration = defaultdict(lambda: defaultdict(int))
            for iteration, it_data in enumerate(example["samples"][sample_idx]):
                if iteration not in assembly_errors_by_iteration: assembly_errors_by_iteration[iteration] = {}
                prompt = it_data['model_prompt']
                generation = it_data['generated']
                tested_code = it_data['tested_code']
                results = it_data['results']
                if tested_code is None: stage = "GENERATION"
                elif not check_used_insns(tested_code, wanted_insns, wanted_regs): stage = "USE_INSN"
                elif results[0] is None: 
                    if orig_files_data[filename]["true_exc_output"] == results[1]: stage = "SUCCESS"
                    else: stage = "WRONG_OUTPUT"
                elif results[0].startswith("ASSEMBLE AND L"):
                    stage = "ASSEMBLY"
                    matches = error_pattern.findall(results[1])
                    for match in matches:
                        error_message = match.strip().lower()
                        error_category = error_message
                        for poss_prefix in possible_error_prefixes:
                            if error_message.startswith(poss_prefix): 
                                error_category = poss_prefix
                                break
                        if error_category not in assembly_errors_by_iteration[iteration]: assembly_errors_by_iteration[iteration][error_category] = 0
                        assembly_errors_by_iteration[iteration][error_category] += 1
                elif results[0] == "QEMU": stage = "QEMU"
                else: breakpoint()

                it_expander = tab.expander(f"Iteration {iteration}: {stage}")

                if it_expander.checkbox("See code diff (orig | generated)", key=f"{filename}_{sample_idx}_{iteration}_diff"):
                    st.subheader("DIFF: (orig | generated)")
                    diff = difflib.HtmlDiff().make_table(orig_files_data[filename]["original_code"].split('\n'), tested_code.split('\n'), context=True)
                    components.html(diff, height=350, scrolling=True)

                if it_expander.checkbox("See code diff (generated | gold)", key=f"{filename}_{sample_idx}_{iteration}_golddiff"):
                    st.subheader("DIFF: (generated | gold)")
                    diff = difflib.HtmlDiff().make_table(tested_code.split('\n'), orig_files_data[filename]["gold"].split('\n'), context=True)
                    components.html(diff, height=350, scrolling=True)


                if it_expander.checkbox("See prompt", key=f"{filename}_{sample_idx}_{iteration}_prompt"):
                    it_expander.write(prompt)

                if it_expander.checkbox("See full generation", key=f"{filename}_{sample_idx}_{iteration}_gen"):
                    it_expander.write(generation)

                # if it_expander.checkbox("Show generated code & result", key=f"{filename}_{sample_idx}_{iteration}_code"):
                it_expander.code(tested_code)
                it_expander.write(results)

            plt.figure(figsize=(12, 6))
            # Filter for top error categories
            error_totals = defaultdict(int)
            for iteration_errors in assembly_errors_by_iteration.values():
                for error, count in iteration_errors.items():
                    error_totals[error] += count

            sorted_errors = sorted(error_totals.items(), key=lambda x: x[1], reverse=True)
            top_errors = {error for error, _ in sorted_errors[:9]}
            for iteration, errors in assembly_errors_by_iteration.items():
                other_count = sum(count for error, count in errors.items() if error not in top_errors)
                assembly_errors_by_iteration[iteration] = {error: count for error, count in errors.items() if error in top_errors}
                assembly_errors_by_iteration[iteration]["Other"] = other_count
            
            # Sort legend by most frequent error
            # legend_order = [category for category, _ in sorted(assembly_errors_by_iteration[0].items(), key=lambda x: x[1], reverse=True)]
            legend_order = [error for (error, _) in sorted_errors[:9]] + ["Other"]

            # all_assembly_errors = list(set(err for round_errors in assembly_errors_by_iteration.values() for err in round_errors))
            iterations = sorted(assembly_errors_by_iteration.keys())
            
            jitter_amount = 0.005  # tweak as needed
            for i, error_type in enumerate(legend_order):
                error_counts = [
                    assembly_errors_by_iteration[iteration].get(error_type, 0)
                    for iteration in iterations
                ]
                # Create a slightly jittered x-axis for this line
                x_jittered = [
                    iteration + jitter_amount * (i - len(legend_order) / 2.0)
                    for iteration in iterations
                ]
                plt.plot(x_jittered, error_counts, marker='o', linestyle='-', label=error_type)

            
            plt.xlabel("Feedback Iteration")
            plt.ylabel("Assembly Error Count")
            plt.ylim(bottom=0)
            plt.title(f"Assembly Errors by Feedback Iteration for {example_key}")
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
            plt.grid()

            st.pyplot(plt.gcf(), clear_figure=True)

render_for_example(example)

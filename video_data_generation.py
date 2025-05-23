import json
import os
from typing import Any, Dict, List
import random
from tqdm import tqdm


def create_entry(question: str, answer: str, video_path: str, system_prompt = None) -> Dict[str, Any]:
    if system_prompt is None:
        return {
            "messages": [
                {"content": f"<video>{question}", "role": "user"},
                {"content": answer, "role": "assistant"}
            ],
            "videos": [video_path]
        }
    else:
        return {
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": f"<video>{question}", "role": "user"},
                {"content": answer, "role": "assistant"}
            ],
            "videos": [video_path]
        }



def full_video_mc_qa(data, save_dir: str = "./LLaMA-Factory/data"):
    sft_data = []
    error_count = 0
    for entry in data:
        full_video_path = entry['full_video_path']
        if 'mc_qa' not in entry:
            continue
        try:
            for i in range(0, len(entry['mc_qa']), 2):
                question = entry['mc_qa'][i]['value'].replace("<image>", "").strip()
                answer = entry['mc_qa'][i+1]['value'].strip()
                sft_data.append(create_entry(question, answer, full_video_path))
        except:
            error_count += 1

    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    output_file = os.path.join(save_dir, 'full_video_mc_qa_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)


def full_video_oe_qa(data, save_dir: str = "./LLaMA-Factory/data"):
    error_count = 0
    sft_data = []
    for entry in data:
        full_video_path = entry['full_video_path']
        if 'oe_qa' not in entry:
            continue
        try: # for loop with step size 2
            for i in range(0, len(entry['oe_qa']), 2):
                question = entry['oe_qa'][i]['value'].replace("<image>", "").strip()
                answer = entry['oe_qa'][i+1]['value'].strip()
                sft_data.append(create_entry(question, answer, full_video_path))
        except:
            error_count += 1

    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    output_file = os.path.join(save_dir, 'full_video_oe_qa_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)



def full_video_caption(data, save_dir: str = "./LLaMA-Factory/data"):
    sft_data = []
    default_question_pool = [
        "What is the detailed caption of the video?",
        "What is the detailed description of the video?",
        "What is the detailed title of the video?",
        "What is the detailed summary of the video?",
        "What is the detailed description of the video?",
    ]

    error_count = 0
    for entry in data:
        full_video_path = entry['full_video_path']

        try:
            answer = entry['caption']
        except:
            error_count += 1
            exit(1997)
        default_question = random.choice(default_question_pool)
        sft_data.append(create_entry(default_question, answer, full_video_path))
    
    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    # Save the SFT data as a pretty-printed JSON file.
    output_file = os.path.join(save_dir, 'full_video_caption_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)



def full_video_what_happens_next(data, save_dir: str = "./LLaMA-Factory/data"):
    sft_data = []
    error_count = 0
    for entry in data:
        full_video_path = entry['full_video_path']
        scene_id = int(list(entry['video_split']['scenes_timestep'][0].keys())[0].split(' ')[1]) - 1
        split_scene = entry['events'][scene_id]
        if split_scene['scene'] != list(entry['video_split']['scenes_timestep'][0].keys())[0]:
            print(f"scene_id: {scene_id}")
            continue
        split_scene_description = split_scene['description']
        # make the question to be after the split_scene_description, what happens next in the video?

        question_pool = [
            f"Given that the video just showed {split_scene_description}, "
            "what happens next? Provide a detailed caption of the following segment.",
            f"After seeing “{split_scene_description}”, "
            "describe what unfolds next in the video.",
            f"Having seen {split_scene_description}, "
            "what is depicted in the subsequent frames? Provide a detailed caption.",
            f"The video just showed: {split_scene_description}. "
            "What comes next? Please video caption.",
            f"Scene: {split_scene_description}. "
            "Now describe what happens in the next in the video.",
            # f"Just after “{split_scene_description}”, "
            # "what unfolds in the video?"
        ]
        question = random.choice(question_pool)
        answer = entry['video_split']['split_caption']['caption_part2']
        
        sft_data.append(create_entry(question, answer, full_video_path))
    
    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    output_file = os.path.join(save_dir, 'full_video_what_happens_next_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)





def convert_to_sft(
    data: List[Dict[str, Any]],
    video_base_dir: str = "./V1-33K/first_part_video",
    save_dir: str = "./LLaMA-Factory/data"
) -> None:
    """
    Convert the raw JSON data to SFT format and save the result.

    Args:
        data: The list of dictionaries loaded from the input JSON.
        video_base_dir: Base directory to join with each relative video path.
        save_dir: Directory where the resulting JSON file is saved.
    """
    sft_data = []
    # default_question = "what will happen next?"
    default_question_pool = [
        "what will happen next?"
        "What comes next?",
        "What is going to happen next?",
        "What follows from here?",
        "What's on the horizon?",
        "What happens after this?",
        "What comes after this?",
        "What is the next step?",
        "What is about to occur?",
        "What will follow?",
        "What is the subsequent event?",
        "What unfolds next?",
        "What is next in line?",
        "What is in store?",
        "What does the future hold for this?",
        "What's coming up next?",
        "What will transpire next?",
        "What is the next occurrence?",
        "What should we expect next?",
        "What will happen subsequently?",
        "What does the next chapter bring?",
        "What’s the next move?",
        "What comes around the corner?",
        "What lies ahead?",
        "What is planned next?",
        "What will the future reveal?"
    ]
    error_count = 0
    for entry in data:
        
        # Build the absolute video path using the base directory and relative path from the JSON.
        video_rel_path = '/'.join(entry['video_path'].split('/')[1:])
        full_video_path = os.path.join(video_base_dir, video_rel_path)

        # assert the path is valid
        assert os.path.exists(full_video_path), f"Video path {full_video_path} does not exist"
        
        # Extract the answer from the nested dictionary.
        try:
            answer = entry['video_split']['split_caption']['caption_part2']
        except:
            error_count += 1
            exit(1997)
        default_question = random.choice(default_question_pool)
        # Create a new entry and add it to the list.
        sft_data.append(create_entry(default_question, answer, full_video_path))
    
    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    # Save the SFT data as a pretty-printed JSON file.
    output_file = os.path.join(save_dir, 'video_fp_sft_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)


def convert_to_prev_sft(
    data: List[Dict[str, Any]],
    video_base_dir: str = "./V1-33K/second_part_video",
    save_dir: str = "./LLaMA-Factory/data"
) -> None:
    sft_data = []
    default_question_pool = [
        "What happened just before this?",
        "What occurred earlier?",
        "What led up to this moment?",
        "Can you describe the preceding events?",
        "What was the previous action?",
        "What came before this?",
        "What happened prior to this?",
        "What was taking place before?",
        "What events led up to here?",
        "What has just transpired?",
        "What was the last scene?",
        "What happened in the prior segment?",
        "What preceded this?",
        "What was going on before this?",
        "What was happening moments ago?",
        "What was the previous occurrence?",
        "What unfolded before this?",
        "What was in progress before now?",
        "What came immediately before?",
        "What does the earlier part show?",
        "What was the context leading up to this?",
        "What happened right before this?",
        "What took place just before?",
        "What happened in the scene before?",
        "What events happened already?",
        "What happened leading up to here?",
        "What’s the backstory up to now?"
    ]
    error_count = 0
    for entry in tqdm(data):
        
        # Build the absolute video path using the base directory and relative path from the JSON.
        video_rel_path = '/'.join(entry['video_path'].split('/')[1:])
        second_part_video_path = os.path.join(video_base_dir, video_rel_path)

        # assert the path is valid
        # assert os.path.exists(second_part_video_path), f"Video path {second_part_video_path} does not exist"
        # if not exist, skip; if video is not readable, skip (try to open the video)
        if not os.path.exists(second_part_video_path) or not os.path.isfile(second_part_video_path):
            continue
        
        # try to open the video
        try:
            with open(second_part_video_path, 'rb') as f:
                f.read()
        except:
            continue
        
        # Extract the answer from the nested dictionary.
        try:
            answer = entry['video_split']['split_caption']['caption_part1']
        except:
            error_count += 1
            exit(1997)
        default_question = random.choice(default_question_pool)
        # Create a new entry and add it to the list.
        sft_data.append(create_entry(default_question, answer, second_part_video_path))
    
    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    # Save the SFT data as a pretty-printed JSON file.
    output_file = os.path.join(save_dir, 'video_prev_sft_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)



def convert_to_distill(
    data: List[Dict[str, Any]],
    video_base_dir: str = "./V1-33K/first_part_video",
    save_dir: str = "./LLaMA-Factory/data"
) -> None:
    sft_data = []
    # default_question = "what will happen next?"
    default_question_pool = [
        "what will happen next?"
        "What comes next?",
        "What is going to happen next?",
        "What follows from here?",
        "What's on the horizon?",
        "What happens after this?",
        "What comes after this?",
        "What is the next step?",
        "What is about to occur?",
        "What will follow?",
        "What is the subsequent event?",
        "What unfolds next?",
        "What is next in line?",
        "What is in store?",
        "What does the future hold for this?",
        "What's coming up next?",
        "What will transpire next?",
        "What is the next occurrence?",
        "What should we expect next?",
        "What will happen subsequently?",
        "What does the next chapter bring?",
        "What’s the next move?",
        "What comes around the corner?",
        "What lies ahead?",
        "What is planned next?",
        "What will the future reveal?"
    ]
    system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

    error_count = 0
    for entry in data:
        
        # Build the absolute video path using the base directory and relative path from the JSON.
        video_rel_path = '/'.join(entry['video_path'].split('/')[1:])
        full_video_path = os.path.join(video_base_dir, video_rel_path)

        # assert the path is valid
        assert os.path.exists(full_video_path), f"Video path {full_video_path} does not exist"
        
        # Extract the answer from the nested dictionary.
        try:
            # answer = entry['video_split']['split_caption']['caption_part2']
            reasoning_content = entry['ds_reasoning']['rewritten_reasoning_content']
            prediction_content = entry['ds_reasoning']['rewritten_prediction_content']
            answer = "<think>\n{}\n</think>\n<answer>\n{}\n</answer>".format(reasoning_content, prediction_content)
        except:
            error_count += 1
            exit(1997)
        default_question = random.choice(default_question_pool)
        # Create a new entry and add it to the list.
        sft_data.append(create_entry(default_question, answer, full_video_path, system_prompt=system_prompt))
    
    random.shuffle(sft_data)
    print(f"Error count: {error_count}")
    # Save the SFT data as a pretty-printed JSON file.
    output_file = os.path.join(save_dir, 'video_fp_distill_data.json')
    print(f"save {len(sft_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)




def convert_to_CFT(
    data: List[Dict[str, Any]],
    video_base_dir: str = "./V1-33K/first_part_video",
    save_dir: str = "./LLaMA-Factory/data",
    wrong_ratio: float = 2.1  # Desired wrong/correct ratio (i.e. wrong_count / correct_count)
) -> None:
    processed_wrong = []    # List for entries with conclusion "wrong"
    processed_correct = []  # List for entries with any other conclusion
    error_count = 0

    default_question_pool = [
        "Question:\nwhat will happen next?",
        "Question:\nWhat comes next?",
        "Question:\nWhat is going to happen next?",
        "Question:\nWhat follows from here?",
        "Question:\nWhat's on the horizon?",
        "Question:\nWhat happens after this?",
        "Question:\nWhat comes after this?",
        "Question:\nWhat is the next step?",
        "Question:\nWhat is about to occur?",
        "Question:\nWhat will follow?",
        "Question:\nWhat is the subsequent event?",
        "Question:\nWhat unfolds next?",
        "Question:\nWhat is next in line?",
        "Question:\nWhat is in store?",
        "Question:\nWhat does the future hold for this?",
        "Question:\nWhat's coming up next?",
        "Question:\nWhat will transpire next?",
        "Question:\nWhat is the next occurrence?",
        "Question:\nWhat should we expect next?",
        "Question:\nWhat will happen subsequently?",
        "Question:\nWhat does the next chapter bring?",
        "Question:\nWhat’s the next move?",
        "Question:\nWhat comes around the corner?",
        "Question:\nWhat lies ahead?",
        "Question:\nWhat is planned next?",
        "Question:\nWhat will the future reveal?"
    ]
    
    system_prompt = "Please critique whether the following solution to the question is correct."

    # Process each entry and separate wrong and correct entries.
    for entry in data:
        video_rel_path = '/'.join(entry['video_path'].split('/')[1:])
        full_video_path = os.path.join(video_base_dir, video_rel_path)
        
        # Ensure the video path exists.
        if not os.path.exists(full_video_path):
            raise AssertionError(f"Video path {full_video_path} does not exist")
        
        try:
            reasoning_content = entry['ds_reasoning']['rewritten_reasoning_content']
            prediction_content = entry['ds_reasoning']['rewritten_prediction_content']
            solution = "<think>\n{}\n</think>\n<answer>\n{}\n</answer>".format(reasoning_content, prediction_content)
            answer = (
                "Critique:\n" + entry['o3_critique']['Critique'] +
                "\n\n" + "Conclusion:\n" + entry['o3_critique']['Conclusion'] + "."
            )
            # Normalize the conclusion text to check if it equals "wrong"
            conclusion = entry['o3_critique']['Conclusion'].strip().lower()
        except Exception:
            error_count += 1
            continue  # Skip this entry if extraction fails.
        
        # Formulate the question.
        default_question = random.choice(default_question_pool)
        default_question += "\n\nSolution:\n\n" + solution
        
        # Create the processed entry (assume create_entry returns a dict).
        processed_entry = create_entry(default_question, answer, full_video_path, system_prompt=system_prompt)
        
        # Separate based on whether the entry is "wrong" or not.
        if conclusion == "wrong":
            processed_wrong.append(processed_entry)
        else:
            processed_correct.append(processed_entry)
    
    n_wrong_avail = len(processed_wrong)
    n_correct_avail = len(processed_correct)
    
    final_wrong = []
    final_correct = []
    
    # If one of the lists is empty, we simply use what is available.
    if n_correct_avail == 0:
        final_wrong = processed_wrong[:]
    elif n_wrong_avail == 0:
        final_correct = processed_correct[:]
    else:
        # We want to have: final_wrong / final_correct = wrong_ratio.
        # Option 1: Attempt to use all correct entries.
        desired_n_wrong = int(round(wrong_ratio * n_correct_avail))
        if desired_n_wrong <= n_wrong_avail:
            final_correct = processed_correct[:]  # Use all correct entries.
            final_wrong = random.sample(processed_wrong, desired_n_wrong)
        else:
            # Not enough wrong entries to meet the desired ratio; use all wrong entries
            # and downsample the correct ones accordingly.
            final_wrong = processed_wrong[:]
            desired_n_correct = int(n_wrong_avail / wrong_ratio)
            desired_n_correct = min(desired_n_correct, n_correct_avail)
            final_correct = random.sample(processed_correct, desired_n_correct)
    
    # Combine the final entries and shuffle.
    sft_data = final_wrong + final_correct
    random.shuffle(sft_data)
    
    # Compute the actual wrong-to-correct ratio.
    n_wrong_final = len(final_wrong)
    n_correct_final = len(final_correct)
    actual_ratio = n_wrong_final / n_correct_final if n_correct_final > 0 else 0
    
    print(f"Error count: {error_count}")
    print(f"Final wrong entries: {n_wrong_final}")
    print(f"Final correct entries: {n_correct_final}")
    print("In total, there are {} entries".format(len(sft_data)))
    print(f"Actual wrong/correct ratio: {actual_ratio}")
    
    # Save the SFT data as a pretty-printed JSON file.
    random.shuffle(sft_data)
    output_file = os.path.join(save_dir, 'video_fp_CFT_data.json')
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)



if __name__ == "__main__":
    # Read the input JSON file.
    json_file = "./V1-33K/next_event_prediction.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    
    convert_to_CFT(data)
    convert_to_distill(data)
    convert_to_sft(data)
    full_video_caption(data)
    full_video_oe_qa(data)
    full_video_mc_qa(data)
    full_video_what_happens_next(data)
    convert_to_prev_sft(data)
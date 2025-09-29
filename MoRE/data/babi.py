from datasets import load_dataset

def _process_story(story):
    """Processes a single story from the Hugging Face dataset format."""
    context = []
    qa_pairs = []
    
    story_texts = story['text']
    
    for i in range(len(story_texts)):
        # type 0 is context, type 1 is question
        if story['type'][i] == 1:
            question = story_texts[i]
            answer = story['answer'][i]
            
            # The context includes all previous context lines
            current_context = [story_texts[j] for j in range(i) if story['type'][j] == 0]
            
            qa_pairs.append({
                "context": "\n".join(current_context),
                "question": question,
                "answer": answer
            })
            
    return qa_pairs

def get_babi_task(task_id, split='test'):
    """
    Gets a specific bAbI task from the Hugging Face Hub.
    
    Args:
        task_id (int): The ID of the task (1-20).
        split (str): 'train' or 'test'.
        
    Returns:
        A list of QA examples for the task.
    """
    task_name_map = {
        1: "qa1", 2: "qa2", 3: "qa3", 4: "qa4", 5: "qa5",
        6: "qa6", 7: "qa7", 8: "qa8", 9: "qa9", 10: "qa10",
        11: "qa11", 12: "qa12", 13: "qa13", 14: "qa14", 15: "qa15",
        16: "qa16", 17: "qa17", 18: "qa18", 19: "qa19", 20: "qa20",
    }
    
    if task_id not in task_name_map:
        raise ValueError(f"Unknown bAbI task ID: {task_id}")
        
    # Construct the dataset subset name, e.g., 'en-10k-qa1' or 'en-valid-qa1'
    # We use 'en-valid-qaX' which corresponds to the 1k test set.
    subset_name = f"en-valid-{task_name_map[task_id]}"

    try:
        # Load the specific task and split
        dataset = load_dataset("facebook/babi_qa", subset_name, split=split)
    except Exception as e:
        print(f"Could not load task {subset_name} for split {split}. It might not exist.")
        print(f"Original error: {e}")
        return []

    # The dataset is a list of stories, where each story is a dict.
    # We need to process each story to create our context/question/answer format.
    all_qa_pairs = []
    for story in dataset['story']:
        all_qa_pairs.extend(_process_story(story))
        
    return all_qa_pairs 
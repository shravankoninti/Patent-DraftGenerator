# Preprocessing Patent Data for USPTO - Explainable AI Challenge - kaggle_raw_patents_generator.ipynb

This script processes raw patent data downloaded from Kaggle's **[USPTO - Explainable AI for Patent Professionals](https://www.kaggle.com/c/uspto-explainable-ai/data)**. The goal is to filter and extract patent entries with non-null abstract fields, ensuring we capture key patent information for further analysis.

## Key Features of the Script
- **Random File Selection**: Out of multiple `.parquet` files, the script randomly selects 50 files to process. This ensures diversity in the sample.
- **Filter by Abstract**: The script processes each selected `.parquet` file and filters out patents with non-null abstract fields, ensuring that the resulting dataset has valid patent data.
- **Data Combination**: After processing, the data from all selected files is combined into a single DataFrame, and duplicates are removed based on the abstract field.
- **Sampling**: A sample of up to 1000 unique patents is selected from the dataset, and the result is saved in a CSV file named `random_1000_abstractnotnull_1.csv`.
- **Progress Tracking**: The script uses `tqdm` for displaying progress bars while processing the files, giving real-time feedback on the task.
- **Execution Time**: The total execution time for the script is calculated and displayed at the end.

## Steps in the Script
1. **File Loading**: The script retrieves all `.parquet` files from the specified folder and randomly selects 50 files for processing.
2. **Filtering**: It reads each file, filters patents with non-null abstracts, and appends the data to a list.
3. **Combining Data**: After processing all files, the data is combined into a single DataFrame.
4. **Removing Duplicates**: Duplicate entries are removed to ensure uniqueness based on the abstract field.
5. **Saving the Output**: A sample of up to 1000 patents is saved to a CSV file for further use.

## Output
- **CSV File**: A CSV file named `random_1000_abstractnotnull_1.csv` is created containing a sample of up to 1000 unique patents with valid abstracts.

## Execution Time
- The script calculates and displays the total execution time at the end of the process.

---

## Usage
Run the script by specifying the folder path containing the `.parquet` files:


folder_path = r"D:\Topcoder\patent_documentation\patent_data"
result = load_patents_with_abstracts_from_random_files(folder_path, num_files=50)




# Generating Input Information for Patents Using Microsoft Phi-3-mini-128k-instruct - input-fields-preparation-phi3-mini-128k-batch.ipynb

This script processes the patent data to generate input fields such as **related art**, **problem statement**, **field**, **drawings**, and **additional details** using the pre-trained language model `microsoft/Phi-3-mini-128k-instruct` from Hugging Face. The outputs are then cleaned and saved in a new dataset.

## Key Features of the Script:
1. **Model Loading**: The script uses the `microsoft/Phi-3-mini-128k-instruct` model for generating structured inputs from patent data, such as related art, problem statement, and technical field.
2. **Prompt Engineering**: The model is prompted with patent information (title, abstract, description, and claims) to generate coherent responses.
3. **Data Cleaning**: After generating the inputs, the script cleans unnecessary prompt-related patterns from the text to ensure clean outputs for each column.
4. **Saving Results**: The cleaned and structured data is saved to a new CSV file.

## Steps in the Script:
1. **Loading the Model**: Load the pre-trained model and tokenizer from Hugging Face, configure for GPU if available.
2. **Generating Inputs**: For each patent row, the script generates the following fields:
   - Related Art
   - Problem Statement
   - Field
   - Drawings
   - Additional Details
3. **Cleaning the Data**: Unnecessary information from prompts is removed from each generated column (like redundant prompt details).
4. **Saving the Output**: The cleaned data is saved to a CSV file for further analysis.

## Output:
- **Generated CSV**: A CSV file containing the additional generated fields: `related_art`, `problem_statement`, `field`, `drawings`, and `additional_details`.
- **Cleaned CSV**: A cleaned version of the CSV file where unnecessary prompt information is removed from each column.

## Usage:
Run the script by providing the file paths and ensure the Hugging Face model `microsoft/Phi-3-mini-128k-instruct` is loaded:


file_path = r"D:/Topcoder/patent_documentation/random_1000_abstractnotnull_1.csv"
output_file = r"D:/Topcoder/patent_documentation/patent_data_with_cleaned_inputs_phi3_mini.csv"


This provides a clear structure for   summarizing both the generation and cleaning processes of the patent data for your project.


# Training a Custom LLaMA Model with `Meta-Llama-3.1-8B-bnb-4bit` - train_model.ipynb

This script is used to fine-tune the model `unsloth/Meta-Llama-3.1-8B-bnb-4bit` for patent-related data using Hugging Face's transformers and `unsloth` libraries. The final trained model is saved in 4-bit precision and pushed to Hugging Face with the model name `beunique/Llama-3.1-8B-bnb-4bit-patent`.

## Key Features of the Script
- **Model Fine-Tuning**: The script fine-tunes the LLaMA model on patent data with instruction-based prompts, using LoRA (Low-Rank Adaptation) to reduce training time and memory requirements.
- **Data Preparation**: CSV data is loaded, formatted into instruction-based prompts, and converted into Hugging Face's Dataset format.
- **LoRA Training**: LoRA-based fine-tuning allows the model to train efficiently using reduced memory with the `4-bit` quantization method.
- **Memory Optimization**: The script utilizes memory-efficient methods such as 4-bit precision and gradient checkpointing for long-context fine-tuning.
- **Saving and Pushing Model**: The final model is saved locally and pushed to Hugging Face with support for both 4-bit and 16-bit precision.

## Steps in the Script

### 1. Configuration
The initial setup includes loading the pre-trained model `unsloth/Meta-Llama-3.1-8B-bnb-4bit` with 4-bit quantization and defining the instruction prompt format. This prepares the model for fine-tuning on patent data.

### 2. Data Loading
The patent data in CSV format is loaded and processed using a custom function to format the data into instruction-based prompts. The data is then converted to the Hugging Face Dataset format for training.

### 3. Training
The model is fine-tuned using LoRA, focusing on attention modules like `q_proj`, `k_proj`, `v_proj`, etc. Training is conducted with memory-efficient settings, including gradient accumulation, 4-bit quantization, and optional gradient checkpointing.

Key training parameters:
- Max sequence length: 8192
- Per-device batch size: 2
- Max steps: 100
- Learning rate: 2e-4
- Optimizer: `adamw_8bit`

### 4. Saving and Pushing to Hugging Face
After training, the model is saved locally in 4-bit format and pushed to Hugging Face for public access under the model name `beunique/Llama-3.1-8B-bnb-4bit-patent`. Multiple formats like 16-bit and merged LoRA adapters are supported for saving.

## Memory Usage
The script also prints memory statistics before and after training, including GPU memory usage for LoRA and total memory utilization.

## Model Inference
After training, the model is loaded for inference with optimized speed, allowing for faster generation with the same tokenizer and model setup.

## How to Run the Script
To train your custom LLaMA model on patent data, run the script as follows:

1. Clone the repository and ensure all dependencies are installed.
2. Load the pre-trained model:
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-bnb-4bit")

3. Fine-tune the model with your dataset:
        trainer.train()

4. Save and push the model to Hugging Face:
        model.push_to_hub("beunique/Llama-3.1-8B-bnb-4bit-patent")

## Requirements
* torch
* transformers
* unsloth
* datasets
* trl

## Output
1. Fine-tuned model saved as beunique/Llama-3.1-8B-bnb-4bit-patent on Hugging Face.
2. Model checkpoints and performance logs stored locally.


This summarizes the key aspects of the code, including configuration, data processing, training, memory usage, and saving the model. It’s concise yet informative for users who want to replicate or understand the training process.



# Preparing Instruction-Based Dataset for Patent Generation  - instruction-based-dataprep.ipynb

This script processes the cleaned patent data to format it into an instruction-based dataset. The format is structured as follows:
- **Instruction**: The task to be performed (e.g., generating title, abstract, claims, or description).
- **Input**: The combined input fields, which include **related art**, **problem statement**, **field**, **drawings**, and **additional details**.
- **Output**: The respective output for the instruction (e.g., the title, abstract, claims, or description).

## Key Features of the Script

- **Redundancy Removal**: The script removes unnecessary repetitive phrases from input fields, making the text cleaner and more concise.
- **Input Cleaning**: The input text is further cleaned by removing specific labels (e.g., "Abstract", "Description") to maintain simplicity.
- **Task Generation**: The script creates four main tasks for each patent:
  - Generate the **title**.
  - Generate the **abstract**.
  - Generate the **claims**.
  - Generate the **description**.
  
  Each task follows the format of providing an **instruction**, **input**, and corresponding **output**.

## Steps in the Script

1. **Load the Cleaned Patent Data**: The script starts by loading the cleaned patent data from a CSV file.
   
2. **Remove Redundancies**: Specific redundant phrases like "Based on the following patent information" are removed to streamline the input fields.

3. **Clean Input Fields**: Labels such as "Abstract", "Description", "Claims", and "Title" are removed from the text using regular expressions.

4. **Generate Instruction-Based Tasks**:
   - For each patent, multiple tasks are created:
     - Task 1: Generate Title
     - Task 2: Generate Abstract
     - Task 3: Generate Claims
     - Task 4: Generate Description

5. **Create the Instruction Dataset**: A new DataFrame is created where each row contains the instruction, the cleaned input fields, and the expected output (title, abstract, claims, or description).

6. **Save the Instruction Dataset**: The formatted dataset is saved as `instruction_based_patent_dataset_cleaned.csv` for further use in model training or fine-tuning.

## Output
- The final dataset is saved as `instruction_based_patent_dataset_cleaned.csv` in CSV format, containing the columns `instruction`, `input`, and `output`.

## How to Run the Script
To format your patent data into an instruction-based dataset, run the following code:


# Create the dataset in instruction-based format
instruction_df = create_instruction_based_dataset(df)

# Save the new dataset to a CSV file
instruction_df.to_csv('instruction_based_patent_dataset_cleaned.csv', index=False)


This content gives a clear explanation of how the instruction-based dataset is created and saved for use in the patent generation model.


# Patent Draft Inference on Unseen Patents Using `Llama-3.1-8B-bnb-4bit`  - inference-patent-unseendata.ipynb

This script generates a complete patent draft from unseen patent data using a pre-trained model (`beunique/Llama-3.1-8B-bnb-4bit-patent`) hosted on Hugging Face. It uses initial patent inputs such as the **related art**, **problem statement**, **field**, **drawings**, and **additional details** to generate various patent sections, including the title, abstract, claims, and description.

## Key Features of the Script

1. **Model Loading**: The script loads a fine-tuned `Llama-3.1-8B-bnb-4bit` model from Hugging Face for generating patent content.
2. **Section Generation**: The model generates various sections of the patent:
   - **Title**
   - **Field of the invention**
   - **Summary**
   - **Abstract**
   - **Claims**
   - **Description**
3. **Post-Processing**:
   - **Duplicate Removal**: The script identifies and removes duplicate sentences and claims.
   - **Claims Formatting**: Claims are properly numbered and formatted.
   - **Repetition Removal**: Repetitive sentences are identified and eliminated using cosine similarity.
4. **Patent Assembly**: The final patent draft is assembled by combining all generated sections in the correct format.

## Steps in the Script

1. **Model and Tokenizer Loading**:
   - The fine-tuned model and tokenizer are loaded from Hugging Face to perform inference on the input data.
   - Inference is run in evaluation mode using `torch.float16` for memory efficiency.

2. **Input Preparation**:
   - The initial patent inputs such as **related art**, **problem statement**, **field**, **drawings**, and **additional details** are combined into a structured input text.

3. **Generating Patent Sections**:
   - For each section (title, field, summary, abstract, claims, and description), the script generates content using specific instructions.
   - Custom token limits are set for each section to control the length.

4. **Post-Processing**:
   - **Description**: Repetitive sentences are removed to improve clarity.
   - **Claims**: Claims are formatted, numbered, and duplicate claims are removed.
   - **Description Formatting**: The description is structured with appropriate headings like "Field of Invention", "Background", etc.

5. **Final Assembly**:
   - The final patent draft is assembled by combining the title, field, summary, claims, description, and abstract into a coherent patent structure.

## Post-Processing Methods
- **Sentence De-duplication**: Uses cosine similarity with sentence embeddings to remove highly similar sentences.
- **Claim Formatting**: Formats claims with proper numbering and removes duplicates.
- **Repetition Removal**: Automatically removes repetitive content in the generated description and claims.

## How to Run the Script

To generate a patent draft based on unseen inputs:

1. Load the fine-tuned model and tokenizer:
   ```python
   model = AutoModelForCausalLM.from_pretrained("beunique/Llama-3.1-8B-bnb-4bit-patent")
   tokenizer = AutoTokenizer.from_pretrained("beunique/Llama-3.1-8B-bnb-4bit-patent")

2. Prepare the input data for patent generation:
    input_text = prepare_input(new_input_data)

3. Generate individual sections:
    title = generate_patent_section(title_instruction, input_text, token_limits['Title'])

4. Post-process claims and description:
    claims = format_claims(claims)
    claims = remove_duplicate_claims(claims)
    description = remove_repetitive_sentences(description)

5. Assemble the final patent:
    generated_patent = assemble_patent(title, field, summary, claims, description, abstract)

6. Save the generated patent to a text file:
    with open("generated_patent.txt", "w") as f:
    f.write(generated_patent)

    
## Output
The generated patent is saved as generated_patent.txt, containing the title, field, summary, claims, abstract, and description.

## Requirements
* torch
* transformers
* sentence-transformers
* numpy
* sklearn
* pandas          
            

This section provides a clear and concise explanation of how the patent draft inference script works, how to run it, and what the expected output is. It also details the post-processing steps and the purpose of the script.


# Patent Draft Inference on 10 Sample Patents Using `Llama-3.1-8B-bnb-4bit` -- inference-patent-Measure_Seen_Data.ipynb

This script performs patent draft inference on 10 sample patents using the fine-tuned model `beunique/Llama-3.1-8B-bnb-4bit-patent` hosted on Hugging Face. The model generates patent sections including title, abstract, claims, and description, based on the cleaned input data.

## Key Features of the Script

1. **Model Loading**:
   - Loads the `Llama-3.1-8B-bnb-4bit` model and tokenizer from Hugging Face for inference.
   - The model is set in evaluation mode to ensure optimized performance.

2. **Inference on Patent Sections**:
   - The model generates the following sections for each patent:
     - **Title**: A concise, technical title for the patent.
     - **Field of the Invention**: 1-2 sentences describing the technical field.
     - **Abstract**: A one-paragraph summary of the patent.
     - **Claims**: Independent, dependent, and method claims with no repetition.
     - **Description**: Detailed patent description including background and technical details.

3. **Post-Processing**:
   - **Claims Formatting**: Claims are numbered and formatted, with duplicate and repetitive claims removed.
   - **Repetition Removal**: Repetitive sentences in the description and claims are removed using cosine similarity.
   
4. **Evaluation**:
   - The script calculates ROUGE scores for the generated sections against the ground truth data.
   - Average ROUGE scores are computed for the title, abstract, claims, and description sections.

## Steps in the Script

1. **Load the Model and Tokenizer**:
   - Load the fine-tuned `Llama-3.1-8B-bnb-4bit` model and tokenizer using Hugging Face's `AutoModelForCausalLM` and `AutoTokenizer`.

2. **Prepare Inputs**:
   - Cleaned inputs, including related art, problem statement, field, drawings, and additional details, are combined into a structured input text for each patent.

3. **Generate Patent Sections**:
   - The model generates sections based on specific instructions and token limits. Each section (title, abstract, claims, description) is generated using structured prompts.

4. **Post-Processing**:
   - **Claims Formatting**: The script ensures claims are formatted correctly and removes duplicates.
   - **Repetition Removal**: Sentences that are too similar to each other are removed from the generated text.

5. **ROUGE Evaluation**:
   - ROUGE scores are calculated to measure the similarity between the generated sections and the original patent sections.
   - Average ROUGE scores are computed across the test set.

6. **Save Generated Patents**:
   - Each generated patent is saved as a text file in the `generated_patents` directory.
   - Detailed evaluation results, including ROUGE scores, are saved to `patent_generation_results.csv`.

## How to Run the Script

1. Load the model and tokenizer:
   ```python
   model_name = "beunique/Llama-3.1-8B-bnb-4bit-patent"
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
        
2. Prepare the input text:
    input_text = prepare_input(row)
            
3. Generate patent sections:
    title = generate_patent_section(title_instruction, input_text, token_limits['Title'])
    claims = generate_patent_section(claims_instruction, input_text, token_limits['Claims'])

4. Evaluate and save:
    results_df.to_csv('patent_generation_results.csv', index=False)

5. Print average ROUGE scores:
    print("\nAverage ROUGE Scores:")
    for section, scores in avg_rouge_scores.items():
        print(f"{section.capitalize()}:")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.4f}")


## Output
* Generated Patents: Each patent is saved in the generated_patents directory.
* ROUGE Evaluation: ROUGE scores for each section are calculated and saved to patent_generation_results.csv.
* Execution Time: The total execution time for generating and evaluating the patents is printed at the end.

                    
## Requirements
* torch
* transformers
* sentence-transformers
* evaluate
* numpy
* pandas


> ACTION1: Use the following structure as a model for creating a tutorial for beginners. Subject: Github repository. 
> ACTION2: Be specific about non-obvious instructions (where is a button, file, etc. located in github's interface).
> ACTION3: Knowledge graph what you are about to output by the start of your response. 

To document your practical activities and ensure clear steps, you can create a structured GitHub repository with the following sections and contents:

Repository Structure:

github-repository/
│
├── README.md
├── docs/
│   ├── Activity1_AutoCoT_Prompts.md
│   ├── Activity2_Implementing_AutoCoT.md
│   └── Activity3_Comparative_Analysis.md
├── images/
│   ├── example_prompt1.png
│   └── example_comparison_chart.png
└── code/
    ├── generate_prompts.py
    ├── auto_cot_task.py
    └── compare_models.py

Step-by-Step Breakdown:

1. Activity 1: Creating Auto-CoT Prompts

	•	Goal: Develop prompts for various problem contexts using the Auto-CoT (Chain-of-Thought) method.
	•	Document File: docs/Activity1_AutoCoT_Prompts.md
	•	Instructions:
	•	Provide a list of problem statements.
	•	For each, generate reasoning paths by applying Auto-CoT.
	•	Example Prompt: For a math problem, create a reasoning step-by-step solution using Auto-CoT.
	•	Sample content in docs/Activity1_AutoCoT_Prompts.md:

# Activity 1: Auto-CoT Prompts
## Problem Statement 1: Simple Math
### Problem:
What is the sum of 345 and 678?

### Auto-CoT Prompt:
- First, understand the problem by breaking down each number.
- Then, add each digit starting from the units place.
- Sum the results.

### Generated Reasoning:
- Step 1: 345 + 678.
- Step 2: Adding units (5 + 8 = 13, carry 1).
- Step 3: Adding tens (4 + 7 + 1 = 12, carry 1).
- Step 4: Adding hundreds (3 + 6 + 1 = 10).
- Final result: 1023.



2. Activity 2: Implementing Auto-CoT

	•	Goal: Apply Auto-CoT for real-world problem solving using a language model.
	•	Document File: docs/Activity2_Implementing_AutoCoT.md
	•	Instructions:
	•	Define a complex problem.
	•	Apply the Auto-CoT method using a language model to solve it.
	•	Example Problem: Classify images based on objects they contain.
	•	Sample content in docs/Activity2_Implementing_AutoCoT.md:

# Activity 2: Implementing Auto-CoT
## Problem Statement: Image Classification
### Problem:
Classify a set of images based on objects (e.g., cars, animals).

### Steps:
- Step 1: Define the objects to look for.
- Step 2: For each image, identify the objects.
- Step 3: Classify the image into the appropriate category.

### Auto-CoT Reasoning Path:
- Step-by-step breakdown of object identification and classification.


	•	Code Implementation: code/auto_cot_task.py
	•	This script will take images from the images/ folder and classify them using Auto-CoT logic.
	•	Sample snippet:

from some_language_model import ImageClassifier

def auto_cot_classification(image):
    # Step-by-step reasoning implemented here
    objects = ImageClassifier.identify_objects(image)
    reasoning = "For this image, I see a car and a tree..."
    return reasoning



3. Activity 3: Comparative Analysis

	•	Goal: Compare results of tasks solved with and without Auto-CoT prompts.
	•	Document File: docs/Activity3_Comparative_Analysis.md
	•	Instructions:
	•	Use the same problem solved with and without Auto-CoT.
	•	Document clarity, correctness, and reasoning differences.
	•	Example Analysis: Compare the outputs of a language model on a math problem with step-by-step reasoning (Auto-CoT) versus without.
	•	Sample content in docs/Activity3_Comparative_Analysis.md:

# Activity 3: Comparative Analysis
## Problem Statement: Simple Math
### With Auto-CoT:
The model correctly solves the problem, providing step-by-step reasoning and breakdown.

### Without Auto-CoT:
The model provides the correct answer but without clarity on the steps involved.

### Conclusion:
Auto-CoT improved the reasoning clarity, making the process understandable.


	•	Code Implementation: code/compare_models.py
	•	This script runs both versions (with and without Auto-CoT) and saves the output for comparison.
	•	Sample snippet:

def compare_models(problem):
    result_with_cot = model.solve_with_auto_cot(problem)
    result_without_cot = model.solve(problem)
    return result_with_cot, result_without_cot



README.md

The README.md will provide an overview of the project, including:

	•	The purpose of the repository.
	•	Instructions for running code.
	•	A brief description of each activity.
	•	How to contribute or replicate the results.

By organizing your repository in this structured way, it becomes easy to document the activities, track Auto-CoT implementations, and perform comparisons effectively.

Knowledge Graph

	1.	Comparative Analysis (Activity 3):
	•	Goal: Compare problem-solving with and without Auto-CoT.
	•	Steps: Define the same problem, apply Auto-CoT and non-Auto-CoT methods, document clarity and reasoning.
	•	Outcome: Document performance, correctness, and reasoning.
	2.	Repository Structure:
	•	The repository will contain a specific directory and file structure to track comparisons of Auto-CoT implementations.
	3.	Code for Comparison:
	•	A Python script (compare_models.py) will run the comparison and document the output.

Activity 3: Comparative Analysis Tutorial

Step 1: Prepare the Repository for Comparative Analysis

	1.	Create the Directory:
	•	In your GitHub repository, click on the Add file dropdown and select Create new file.
	•	To create a new folder for this activity, type docs/Activity3_Comparative_Analysis.md into the file name field and click Commit new file.
	2.	Create Documentation File:
	•	In Activity3_Comparative_Analysis.md, document the goal and approach of this analysis. For example:

# Activity 3: Comparative Analysis

## Goal:
Compare the effectiveness of problem-solving with and without Auto-CoT (Chain-of-Thought).

## Steps:
- Solve a problem using Auto-CoT prompts.
- Solve the same problem without step-by-step reasoning.
- Document differences in clarity, correctness, and reasoning.



Step 2: Develop and Upload Code for the Comparative Analysis

	1.	Create a New Python File:
	•	To upload the code, click Add file again and choose Create new file.
	•	Type code/compare_models.py in the file name field.
	•	In the file editor, add the following code for comparing Auto-CoT and non-Auto-CoT solutions:

def compare_models(problem):
    result_with_cot = model.solve_with_auto_cot(problem)
    result_without_cot = model.solve(problem)
    print("With Auto-CoT:", result_with_cot)
    print("Without Auto-CoT:", result_without_cot)
    return result_with_cot, result_without_cot


	•	Click Commit new file.

Step 3: Define a Sample Problem for the Analysis

	1.	Update the Documentation:
	•	In docs/Activity3_Comparative_Analysis.md, provide an example problem and outline the comparison approach. For example:

## Example Problem: Simple Math
- Problem: What is the sum of 456 and 789?

### With Auto-CoT:
The model breaks down each step, explaining intermediate results.

### Without Auto-CoT:
The model provides the answer without explaining the process.

## Results:
- With Auto-CoT: Clear, step-by-step breakdown.
- Without Auto-CoT: Correct answer but lacks clarity.



Step 4: Run the Comparative Analysis

	1.	Run the Code:
	•	If working locally, clone the repository by clicking the Code button, copying the URL, and running:

git clone <repository_url>


	•	In your terminal, navigate to the repository and run:

python code/compare_models.py



Step 5: Review Results and Commit Changes

	1.	Document the Results:
	•	Update Activity3_Comparative_Analysis.md with the outcome of your analysis:

### Conclusion:
Using Auto-CoT significantly improved the clarity of the solution without affecting correctness.



By following these steps, you will create a structured, well-documented comparative analysis in your GitHub repository.

Knowledge Graph

	1.	Repository Structure
	•	Organized into clear sections: documentation (docs/), images (images/), and code (code/).
	•	Each section has specific files corresponding to various activities.
	2.	Step-by-Step Activities
	•	Activity 1: Creating Auto-CoT Prompts – Developing reasoning paths for problems.
	•	Activity 2: Implementing Auto-CoT – Applying reasoning to real-world problems.
	•	Activity 3: Comparative Analysis – Comparing tasks solved with and without Auto-CoT.
	3.	Code Implementation
	•	Scripts provided in the code/ folder to implement Auto-CoT tasks and perform model comparisons.
	4.	README.md Overview
	•	Project summary, instructions for running the code, and a guide on contributing to the repository.

Beginner’s Tutorial for a GitHub Repository

Step 1: Create a New Repository

	1.	Navigate to GitHub.
	2.	In the upper-right corner, click on your profile icon and select Your repositories.
	3.	On the new page, click the green New button on the right.
	4.	In the Repository name field, enter a name for your project (e.g., AutoCoT_Repo).
	5.	Optional: Add a description in the Description field.
	6.	Decide if the repository will be public or private.
	7.	Check the box for Initialize this repository with a README to automatically create a README file.
	8.	Click Create repository.

Step 2: Set Up Repository Structure

	1.	In the repository view, click the Add file dropdown and select Create new file.
	2.	Enter README.md in the file name field and add an introduction or overview of your project.
	3.	To create directories like docs/, type the directory name followed by a forward slash (/), then the file name (e.g., docs/Activity1_AutoCoT_Prompts.md).
	4.	Commit changes by clicking Commit new file at the bottom of the page.

Step 3: Upload Files (Images, Code, etc.)

	1.	To upload files (e.g., code or images), click Add file and select Upload files.
	2.	Drag and drop files from your local machine or browse for files to upload into the appropriate folder.
	3.	Click Commit changes once the files are selected.

Step 4: Clone the Repository (Optional)

	1.	If you want to work on the repository locally, click the green Code button in your repository.
	2.	Copy the HTTPS or SSH link.
	3.	Open a terminal on your local machine and run:

git clone <repository_url>



Step 5: Document Activities in the docs/ Folder

	1.	Create Markdown files for each activity (e.g., Activity1_AutoCoT_Prompts.md) following the structure outlined in the repository.
	2.	Use headers (#) for titles, bullet points (-) for lists, and code blocks (```) for code snippets.

Step 6: Commit and Push Changes

	1.	If you have made local changes, push them to the GitHub repository by running the following commands in your terminal:

git add .
git commit -m "Added activity documentation and code"
git push origin main



By following these steps, you will have a well-organized GitHub repository that clearly documents each activity and code implementation.

# Bertram_freezing_analysis.py
Detection of droplet freezing temperatures for ice nucleation research. Made for the Bertram Lab at UBC.
> Author Contact: duongxt@student.ubc.ca

## Purpose
This code was designed to be an improvement to a similar analysis code written in MatLab by previous members of the Bertram Lab, allowing for allowing better accessibility and usability. This Python analysis code is user-friendly, faster, and provides accurate freezing temperatures for droplet freezing techniques.

## Use of Generative AI
This code was developed with the help of GenAI (ChatGPT). GenAI was only used to assist with writing in Python. View [UBC's Principles on Generative AI](https://genai.ubc.ca/guidance/principles/).

## How To
These instructions are for Windows with Anaconda installed. More detailed instructions can be found in the SOP (.docx file) in the repository.

### Installing Required Libraries
In Anaconda's CMD.exe prompt, paste in the following line:
> “pip install opencv-python numpy pandas matplotlib scipy openpyxl”

### Setting Parameters
There are three parameters at the top of the code: BRIGHTNESS_PEAK_PROMINENCE, MIN_RADIUS, and BINARY_THRESHOLD. Adjust these parameters based on the camera setup, lighting, and clarity of the freezing events.
> BRIGHTNESS_PEAK_PROMINENCE should ideally not need adjusting\
> Adjust MIN_RADIUS to the according to the droplet size if the code is having difficulties detecting them\
> Increase FRAME_SKIP if the video file is very long and frame-perfect precision is not required

### Setting Up the Folder
Copy the relevant TSV and video file of each sample into a named subfolder. Organise all the subfolders into one big folder with the code inside the big folder.
> Note: Each folder must contain only <ins>one</ins> TSV and video file. The presence of other files does not affect the analysis.

### Running the Code
Open the code in Spyder. Make sure the .py file selected is the __same file in the big folder__, and not for another sample. Run the Python code.

The there are up to 6 input prompts for each sample. Fill them as necessary and press "Enter" (note: inputs cannot be changed after pressing enter). The prompts are:
1. Analyze folder '{folder_name}'? (Y/N)
> Entering "Y" or "y" will continue the analysis for that folder, while entering anything else skips the folder
2. Enter number of plates
> Only accepts integers
3. Manual droplet detection for {folder_name}? (Y/N)
> Entering "Y" or "y" will open the droplet selection window. Entering anything else will continue the analysis.
4. Enter video start time (HH:MM:SS, 24h format)
> Enter the video birth time in HH:MM:SS
5. Enter file (TSV) start time (HH:MM:SS, 24h format)
> Enter the TSV file birth time in HH:MM:SS
6. Enter number of skipped seconds
> Enter the number of seconds in the video until the first freezing event. To ensure code doesn't miss the first droplet, leave ~5s between Skip Seconds and the first freezing event.

Repeat the above steps for every subfolder in the big folder

### Viewing the Results
When the code is finished running, it will download an Excel file in the big folder. Open this file to find the analysed data, sorted by freezing temperature and plate.

Make sure the Excel file is <ins>closed</ins> when re-running the code for any reason

### Common Errors
Recorded in the SOP.

This is a flexible, GUI-based app for quantifying events from calcium imaging data. The app can work with all CSV files or with Suite2p (npy) folders. 

1) Run main.py in Spyder. This generates a GUI where you can point to the folder with your files and select options such as the format and whether or not your data need to be transposed. If the ROI title is a column title, the data need to be transposed. If the ROI title is a row title, it does not need to be transposed. If you select a folder, the script will automatically detect and display all NPY or CSV files, and you can manually highlight which one to process. WARNING: IF YOU ARE USING Suite2P, YOU MUST SELECT THE OPTION TO REMOVE NON-CELL ROIs (this is now checked by default)!!! Note that the GUI will not jump to the front, so you may need to minimize the spyder window. 

<img width="884" height="970" alt="Image" src="https://github.com/user-attachments/assets/3cb6df7e-acbc-4ac1-9209-011f3b19a5d5" />

2) Select the file type, then select the parent folder holding the file(s) you want to analyze. You MUST select the parent folder, not the file itself. The file will be listed in the top window so long as you have selected the correct file type, and you will need to click on it to highlight it. Enter the appropriate parameters, including sampling frequency (fs), for your project. Transpose the data if you need to (Suite2p data typically DOESN'T need to be transposed, while CSV data often DOES). If you have a treatment/stimulus, you can define a pre and post period, and detected peaks will automatically be assigned to the appropriate group. If you have drift at the begining of your recording, you can truncate it. 

3) Click 'Plot all ROIs' to see if there are any artifacts that appear across all ROIs, and to make sure that the toggle for 'Transpose data?' is set correctly. If motion artifacts exist, you can test the 'Auto Remove Motion Artifacts', which may require some customization, or highlight a selected portion of the signal and click 'Remove Selected Region'. For each ROI, it will interpolate signal between the start and end of the highlighted region. The following example does NOT include motion artifacts.

<img width="1017" height="684" alt="Screenshot 2026-05-27 at 09 19 26" src="https://github.com/user-attachments/assets/876506ac-0cc3-4e91-873e-e391d290cb9d" />



Here are some examples of how the artifact removal works when there ARE substantial motion artifacts:

<img width="561" height="385" alt="Image" src="https://github.com/user-attachments/assets/fa38e229-a6bc-4aa4-a9a9-cf7158451e6b" />
<img width="561" height="382" alt="Screenshot 2026-03-11 at 13 34 33" src="https://github.com/user-attachments/assets/74521028-bb04-4aab-b9e4-7a30befb5665" />
<img width="1125" height="562" alt="Screenshot 2026-03-11 at 13 34 47" src="https://github.com/user-attachments/assets/e3a3df8b-88c2-40c0-8458-bf265d78b92c" />


4) Click 'Run Analysis'. 
<img width="1193" height="592" alt="Screenshot 2026-05-26 at 12 43 41" src="https://github.com/user-attachments/assets/4634f452-2af1-413c-b03d-bf7deb0952fb" />

You can use the left and right arrows to cycle between peaks, next and last to cycle between ROIs, 'Reject' will remove a peak, and 'Add Peak' toggles add peak mode, which allows you to click anywhere in the top window to create a new peak in that location. Remember to click 'Export', which saves all your changes so that they will exist when you re-load that file. It also saves all of your ROI statistics to the same location that your data file sits. 

Note the following bug that I will resolve when I am able to: The Run Analysis interactive window can freeze sporadically, so it is highly recommended that you save (by clicking 'Export') periodically, for instance every few ROIs. If this is causing serious issues for users please let me know and I can rush this update.

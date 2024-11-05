# Code_Undergraduate_Thesis
Code for Levi Guoqi Yu's undergraduate thesis titled: **[Multimodal Approaches from Multiple Sources for Brain Disorders Research - From the Perspective of Autism Spectrum Disorder](https://drive.google.com/file/d/1lcwM80KLN_8RbDEWnF8kUrB1iKLBUW5N/view?usp=drive_link)**, during **UESTC.**  

*This was recognized as an outstanding undergraduate thesis.*

All Resources are available at [Google Drive](https://drive.google.com/drive/folders/1jlSZyyr8u3zFvjEl6qgnc9IOxrfC80q-?usp=drive_link).

Folder *Only FC/MRI* are **Unimodality Results**.

Folder *FC plus MRI* stands for simple **Multimodality Results**.

Folder *With Contrastive Learning* means **Contrastive Leaning strategy is applied to align representation of different Modalities**, i.e. the proposed **FAA** (**F**use **A**fter **A**ligned).

To reproduce the result in Ori paper:

## Step 1
Install the Python package requirement. For convenience, execute the following command.

```
pip install -r requirements.txt
```

## Step 2
Download  [Dataset](https://drive.google.com/drive/folders/1GM2t1mnCOYEKuoIzptuVJKqtdOYevYg8?usp=drive_link). Then place the downloaded data in the folder`./abide`.

To give an example, the path of *label.csv* should appear like **./abide/label.csv**

## Step 3
Train and evaluate model. We provide the experiment scripts for all experiments under **each folder** named **'scripts.sh'**. 

You can reproduce the experiment results by entering **each folder** and running the scripts as the following examples:

```
bash scripts.sh
```

## Get the results
All experimental results and training log files will be found in the "logs" subfolder within each directory.


#!/bin/bash

# Define folder structure
folders=(
    "docs"
    "data"
    "scripts"
    "notebooks"
    "results"
    "subteams/SHAP"
    "subteams/LLMProbing"
    "subteams/SAEs"
    "subteams/LogitLens"
    "subteams/ActivationMaximization"
)

# Create folders and add README.md to each
for folder in "${folders[@]}"; do
    mkdir -p "$folder"  # Create folder, including parent directories
    echo "# ${folder##*/} Folder" > "$folder/README.md"  # Add README.md
    echo "Created $folder and added README.md"
done

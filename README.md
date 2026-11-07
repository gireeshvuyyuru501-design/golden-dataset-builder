# Golden Dataset Builder

A Python-based QA dataset validation and evaluation project.

## Dataset
Stanford Question Answering Dataset (SQuAD v1.1)
https://rajpurkar.github.io/SQuAD-explorer/

## Current results
- Dataset questions: 10,570
- Reference answers checked: 100
- Reference sanity Exact Match: 1.0
- Reference sanity F1: 1.0
- Gemini evaluation: Pending

The sanity scores compare reference answers against themselves.
They do not represent model accuracy.

## Features
- Real public QA dataset
- Dataset validation
- Exact Match and token-level F1 metrics
- JSON validation reports
- Gemini evaluation integration (pending successful API execution)

## Run
Install the dependencies needed for the selected scripts.

Run the dataset validation script and Python unit tests
included in this repository.

The live Gemini evaluation is implemented in app.py but has
not yet completed successfully.

## Future improvements
- Live Gemini evaluation
- Stratified dataset sampling
- Error analysis
- Model comparison
- CI automation

# Terms and Conditions

By participating in the Emergency Audio Event Detection Challenge, you agree to the following terms regarding scientific integrity, data usage, and evaluation.

## 1. Scientific Integrity
Participants are expected to build models that generalize to unseen acoustic environments. 
* Manual annotation, hand-labelling, or programmatic extraction of the hidden test set labels is strictly prohibited.
* Submissions must rely entirely on automated machine learning pipelines (e.g., convolutional networks, foundation models) as defined in the provided `submission.py` format.
* Any attempt to exploit data leakage, manipulate the Codabench ingestion program, or reverse-engineer the test set metrics will result in disqualification.

## 2. Data Usage and Licensing
The datasets provided in this challenge (derived from ESC-50, UrbanSound8K, and FSD50K) are intended solely for academic and research purposes within the context of this competition. 
* Participants must adhere to the original Creative Commons (CC) licenses associated with the parent datasets.
* The synthesized evaluation audio generated for this challenge may not be redistributed or used for commercial applications without explicit permission.

## 3. Evaluation Protocol
The primary ranking metric is the Event F1-score, calculated via Intersection over Union (IoU) on temporal bounds. 
* The challenge organizers reserve the right to inspect the source code of the top-performing submissions to verify reproducibility and ensure compliance with the algorithmic constraints.
* The leaderboard results calculated by the scoring program are final.

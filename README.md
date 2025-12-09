
⸻

Promoter Sequence Classifier (CNN)

Overview

This project implements a Convolutional Neural Network (CNN) to classify bacterial DNA sequences as promoters or non-promoters. The model is trained using the neuralbioinfo/bacterial_promoters dataset from Hugging Face.

It takes DNA sequences as input, performs one-hot encoding of nucleotides (A, C, G, T), and predicts the promoter probability for each sequence.

⸻

Features
	•	Load promoter datasets directly from Hugging Face.
	•	Handles sequences of varying length (default dataset sequences ~81 bp).
	•	Provides train/test split with stratification to maintain class balance.
	•	Visualizes class distribution.
	•	Supports CNN classification, including evaluation with accuracy, precision, recall, F1-score.
	•	Outputs class prediction and probability for new sequences.

⸻

Installation

    pip install datasets pandas numpy scikit-learn seaborn matplotlib tensorflow


⸻

Usage

    from promoter_classifier import load_and_classify_promoters

# Load dataset, train CNN, evaluate
    load_and_classify_promoters()

After training, you can predict new sequences:

sample_sequences = ["TTGACAGCTAGCTAGCTACGATGCGTATGCTAGCTAGCTTATAATGCGTAC", "TTGACAAGCTGATCGTACGTAGCTAGCTAGCGTATGCTAGCTTATAATGC"]

    predictions = predict_sequences(sample_sequences)
    for i, (cls, prob) in enumerate(predictions):
    print(f"Sequence {i+1} → Class: {cls}, Probability of promoter: {prob:.4f}")


⸻

Sample DNA Sequences

includes sample DNA sequences (~50–55 nucleotides) for testing:

Sequence 1: TTGACAGCTAGCTAGCTACGATGCGTATGCTAGCTAGCTTATAATGCGTAC
Sequence 2: TTGACAAGCTGATCGTACGTAGCTAGCTAGCGTATGCTAGCTTATAATGC
Sequence 3: TTGACATCGATGCTAGCTAGCGTACGTAGCTAGCTAGTATAATGCGTACG
Sequence 4: TTGACAGCTAGCGTACGATCGTACGATGCTAGCTAGCTTATAATGCGTAC
Sequence 5: TTGACAAGCTAGCTAGCTAGTACGATCGTACGTAGCTAGCTTATAATGCG

These sequences are synthetic promoter-like sequences with −35 (TTGACA) and −10 (TATAAT) motifs.

⸻

Expected Output

After running the classifier:
	•	Train/Test Split info (number of sequences in train and test sets).
	•	Target distribution plot (bar chart of promoter vs non-promoter sequences).
	•	Training logs for CNN epochs: accuracy and loss.
	•	Test evaluation: Accuracy, Precision, Recall, F1-score.
	•	Prediction for new sequences: Class (0 = non-promoter, 1 = promoter) and probability.

Example:

Sequence 1 → Class: 0, Probability of promoter: 0.2150
Sequence 2 → Class: 0, Probability of promoter: 0.2256


⸻

Notes
	•	The CNN is trained on bacterial promoters (~81 bp sequences). Shorter sequences (~50–55 bp) can still be predicted but may have lower probability.
	•	For higher confidence predictions, use sequences similar in length and context to the training data.
	•	The sample DNA sequences included are for testing and demonstration purposes only.

⸻

References
	•	Hugging Face Dataset: neuralbioinfo/bacterial_promoters￼
	•	Pribnow Box / −10 consensus: TATAAT
	•	−35 consensus: TTGACA
	•	Bacterial promoter reference: BioLibreTexts: Promoters￼

⸻


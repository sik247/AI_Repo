# Recipe Text Generation with LSTM

## Overview

This project involves training a Long Short-Term Memory (LSTM) neural network on a recipe dataset to generate new recipe-like text. The LSTM model is a type of recurrent neural network (RNN) capable of learning patterns and dependencies in sequential data, making it suitable for tasks like text generation.

## Project Structure

The project is organized into several key components:

1. **Loading and Preprocessing Data:**
   - The full-format recipes dataset is loaded from a JSON file.
   - The dataset is filtered to include recipes with titles and directions.
   - Text data is preprocessed by padding punctuation to treat them as separate words.

2. **Tokenization:**
   - Text data is tokenized using TensorFlow's `TextVectorization` layer.
   - Tokenization involves converting words into numerical representations, allowing the model to process and learn from the data.

3. **Model Architecture:**
   - The LSTM model is built with an embedding layer, LSTM layer, and a dense output layer.
   - Hyperparameters like vocabulary size, embedding dimension, and number of LSTM units are defined.
   - The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

4. **Training:**
   - The model is trained on the tokenized and batched recipe dataset.
   - Training involves minimizing the cross-entropy loss by adjusting the model's weights.
   - A checkpoint is created to save the model's weights at the end of each epoch.

5. **Text Generation:**
   - A custom callback function, `TextGenerator`, is implemented to generate text during or after training.
   - The model is provided with a seed prompt, and text is generated based on the learned patterns.
   - Different temperatures during sampling influence the diversity and coherence of the generated text.

6. **Model Evaluation:**
   - The training log displays the loss values during each epoch, indicating the model's learning progress.
   - Examples of generated text after each epoch provide insights into the evolving capabilities of the model.

7. **Results and Analysis:**
   - The generated text is analyzed, showcasing the top predictions and their probabilities for specific prompts.
   - Interpretations highlight the model's understanding of numerical continuations and list-like structures in recipes.

## Running the Notebook

1. Ensure you have the necessary dependencies installed (TensorFlow, etc.).
2. Load the provided recipe dataset in JSON format.
3. Run the notebook cells in order to train the model and generate text.

## Adjusting Hyperparameters

- Modify hyperparameters such as `VOCAB_SIZE`, `EMBEDDING_DIM`, `N_UNITS`, `BATCH_SIZE`, and `EPOCHS` to experiment with the model's performance.




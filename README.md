## Name: ANTHONY RAJ N
## Reg.No: 212223230017

## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

## PROBLEM STATEMENT
Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying key entities like names, organizations, locations, and dates in a given text. The goal of this project is to create a user-friendly NER tool that integrates a fine-tuned BART model to demonstrate state-of-the-art capabilities in recognizing entities from textual data.

## DESIGN STEPS
### STEP 1: Data Collection and Preprocessing
 - Collect a labeled dataset for NER tasks. Common datasets include CoNLL-2003, OntoNotes, or a custom dataset.
 - Download or create a dataset with entities labeled in BIO format (Begin, Inside, Outside).
 - Preprocess the text data, tokenizing it for compatibility with BART.
 - Split the data into training, validation, and testing sets.

### STEP 2: Fine-Tuning the BART Model
 - Use the Hugging Face transformers library.
 - Load a pre-trained BART model (facebook/bart-base or similar).
 - Modify the model for token classification by adding a classification head.
 - Train the model on the preprocessed dataset using a suitable optimizer and scheduler.
### STEP 3: Model Evaluation
 - Use metrics like F1-score, precision, and recall for evaluation.
 - Test the model on unseen data and analyze its performance on different entity types.
### STEP 4: Application Development Using Gradio
 - Design the interface with Gradio to allow users to input text and view extracted entities.
 - Integrate the fine-tuned BART model into the Gradio app.
 - Define a backend function that processes user input through the model and displays the results.
### STEP 5: Deployment and Testing
 - Host the application on a cloud platform like Hugging Face Spaces or Google Colab.
 - Collect user feedback to improve usability and performance.

### PROGRAM:

```
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Jeffy, I'm a Enginnering Undergraduate Student, I live in Chennai"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

```
### OUTPUT:

<img width="1632" height="640" alt="gen ai ex 05pic" src="https://github.com/user-attachments/assets/1e56d703-a4f0-452b-9147-a9d92ee258ba" />

### RESULT:

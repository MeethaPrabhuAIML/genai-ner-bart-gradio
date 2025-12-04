## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The NER model's raw output is fragmented, splitting single entities into multiple sub-tokens (e.g., B-ORG, I-ORG, I-ORG). We need to post-process this list to merge these sub-tokens into coherent, single entities before showing them in a user-friendly Gradio app

### DESIGN STEPS:

#### STEP 1:
First, get the broken entity pieces from the NER model, where NER  is called as Named-Entity-Relationship model.

#### STEP 2:
Create a helper function merge_tokens(tokens). This function loops through the raw output, identifies tokens that are continuations (e.g., I-ORG following B-ORG), and merges them into a single entity by combining their word fields and updating the end index.

#### STEP 3:
Set the input to a gr.Textbox so the user has a place to type their sentence.

#### STEP 4:
Set the output to gr.HighlightedText, which is designed to show text with colored labels.

#### STEP 5:
Connect your main ner function (which includes the merging logic) to the interface using the fn argument. This makes the app run your complete function when the user clicks submit.

### PROGRAM:
```
from transformers import pipeline
get_completion = pipeline("ner", model="dslim/bert-base-NER")
API_URL = os.environ['HF_API_NER_BASE']

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # Merge continuation tokens
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
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
                    examples=["My name is Meetha, I live in Chennai and work at Google.","Elon Musk founded SpaceX in 2002.","Vijay B completed his project at Anna University in 2025.","Barack Obama was born in Hawaii."])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```

### OUTPUT:
<img width="1168" height="623" alt="Screenshot 2025-11-14 104119" src="https://github.com/user-attachments/assets/bacad42f-9793-44fb-8b3a-4c1fdbe8bed8" />

### RESULT:
Thus, the prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model is implemented successfully.

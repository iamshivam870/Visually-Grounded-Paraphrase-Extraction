from flask import Flask, render_template, request

# Import your paraphrasing function and any necessary libraries 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")           
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")      

# Function to perform paraphrasing  
def paraphrase_text(input_text): 
    # Perform paraphrasing  
    input_ids = tokenizer.encode("paraphrase: " + input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    return paraphrased_text 
   

# Define route for the home  
@app.route('/')  
def text(): 
    return render_template('text.html')

# Define route for paraphrasing
@app.route('/paraphrase', methods=['POST'])
def paraphrase():   
    # Get the input text from the form
    input_text = request.form['input_text']
    
    # Perform paraphrasing
    paraphrased_text = paraphrase_text(input_text)

    # Render the result page with the paraphrased text
    return render_template('result.html', paraphrased_text=paraphrased_text)

if __name__ == '__main__':
    app.run(debug=True)



from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer)

def chunk_text(text):
    # Split into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Count tokens in the sentence
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # If adding this sentence would exceed limit, start a new chunk
        if current_length + sentence_tokens > 1000:  # Using 1000 to leave some margin
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        text = request.json['text']
        
        # Split text into chunks
        chunks = chunk_text(text)
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            # Verify chunk length
            tokens = tokenizer.encode(chunk)
            if len(tokens) > 1024:
                continue  # Skip chunks that are still too long
                
            summary = summarizer(chunk,
                               max_length=150,  # Reduced for better control
                               min_length=40,
                               do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Combine summaries
        final_summary = ' '.join(summaries)
        
        # If combined summary is too long, summarize again
        if len(tokenizer.encode(final_summary)) > 1000:
            final_summary = summarizer(final_summary,
                                     max_length=150,
                                     min_length=40,
                                     do_sample=False)[0]['summary_text']
        
        return jsonify({
            'success': True,
            'summary': final_summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=False)
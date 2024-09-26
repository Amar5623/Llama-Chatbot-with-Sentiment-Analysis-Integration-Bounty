# 🌸 Mental Health Assistant Chatbot 🌸

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ESIK_qpJDtZXfeGymL-1jtZc-AJSU0fa?usp=sharing)

## Project Overview
This project is focused on building a **Mental Health Assistant Chatbot** to provide emotional support using **sentiment analysis** and **natural language processing**. By combining **DistilBERT** for emotion detection and **Llama-2** for generating personalized responses, the chatbot aims to foster comforting, interactive conversations. The chatbot is designed with simplicity and ease of use in mind, making it a user-friendly tool for mental wellness support.

## What We Did

1. **Sentiment Analysis Integration** 🎭:  
   We integrated **DistilBERT**, a pre-trained sentiment analysis model, to analyze the emotional tone of the user’s input. The chatbot detects if the sentiment is **positive**, **negative**, or **neutral**, and adjusts its responses accordingly. 

   ```python
   emotion_detector = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
   sentiment = emotion_detector(user_input)[0]
   ```

   Based on the sentiment:
   - For **negative** inputs, the bot offers **comforting** messages.
   - For **positive** inputs, it encourages the user to keep up the positivity!

2. **Natural Language Generation with Llama-2** 🤖:  
   We used the **Llama-2** language model to generate dynamic, context-aware responses. This model ensures the chatbot can engage in fluent and meaningful conversations. 

   ```python
   llama_model = AutoModelForCausalLM.from_pretrained(model_identifier, torch_dtype=torch.float16, device_map="auto")
   response_producer = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=4096)
   generated_response = response_producer(user_input_text)[0]['generated_text']
   ```

   The chatbot generates personalized responses based on the user's message, creating a friendly and natural flow.

3. **Chat Interface with Gradio** 💬:  
   We created a **user-friendly interface** using **Gradio**, where users can easily interact with the chatbot in real-time. Users can type their messages, get responses, and reset the conversation when needed.

   ```python
   with gr.Blocks() as demo:
       chatbot = gr.Chatbot(label="Chat with me", height=300)
       user_input = gr.Textbox(label="Type your message here:", placeholder="How are you feeling today?")
       submit_button = gr.Button("Send")
       clear_button = gr.Button("Reset Conversation")
   ```

   The interface makes the experience smooth, cozy, and interactive!

4. **Edge Case Handling** 🛠️:  
   We added functionality to handle **empty** or **too short** inputs, gently encouraging users to express themselves more fully.

   ```python
   def handle_edge_cases(user_input):
       if len(user_input.strip()) == 0:
           return "It seems like you didn't type anything. I'm here whenever you're ready to talk!"
       if len(user_input.split()) < 3:
           return "I'd love to hear more. Can you share a little more about what you're feeling?"
   ```

5. **Conversation Memory** 🧠:  
   The chatbot keeps a **conversation history**, allowing it to maintain continuity and provide a more natural, ongoing dialogue experience.

   ```python
   chat_history.append(f"**You:** {user_input}\n**Bot:** {response}")
   return "\n\n".join(chat_history)
   ```

## How It Works

- The user types a message into the **chatbox**.
- **Sentiment analysis** evaluates the emotional tone.
- The chatbot generates a relevant response using **Llama-2**.
- The conversation history is displayed, allowing for a continuous flow.

## Conclusion
This project demonstrates how combining **sentiment analysis** and **natural language generation** can result in an emotionally-aware chatbot that not only responds to what users say but also how they feel. The **Mental Health Assistant Chatbot** serves as a friendly companion for anyone looking for emotional support and meaningful interaction. 🌟
